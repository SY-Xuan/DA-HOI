import json
import torch
import argparse
from qwenvl.data.hico_class import *
import numpy as np
import re
from hico_eval import HICOEvaluator


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


sentence2action_object = {}

for interaction in HICO_INTERACTIONS:
    if interaction['action'] == "no_interaction":
        continue
    else:
        act = interaction['action'].split("_")
        act[0] = VERB_MAPPER[act[0]]
        act = " ".join(act)
        s = f"{act} a {interaction['object']}"
        sentence2action_object[s] = (interaction['action'], interaction['object'])

def extract_object_action(prediction):
    return sentence2action_object[prediction]

def extract_object_action_bbox(prediction):
    bbox_pattern = re.compile(r'[0-9].[0-9][0-9]')
    first_part, second_part = prediction.split(": ")

    action_name, object_name = extract_object_action(first_part)
    pred_hoi_id = HICO_NAME2index[(action_name, object_name)]

    res = bbox_pattern.findall(second_part)
    pred_person_bbox = []
    pred_object_bbox = []
    assert len(res) == 8
    for index, r in enumerate(res[:4]):
        pred_person_bbox.append(float(r))
    for index, r in enumerate(res[4:]):
        pred_object_bbox.append(float(r))
    return pred_hoi_id, pred_person_bbox, pred_object_bbox


def parse_prediction_group(results, predictions):

    sub_bbox = []
    obj_bbox = []
    verb_score = []
    obj_label = []

    format_error = 0
    idx = predictions['img_id']
    for object_name, bbox, scores, potential_list, pred_logits in zip(predictions['pred_object_names'], predictions['bboxes'], predictions['max_scores'], predictions['merge_potential_list'], predictions['merge_pred_logits']):
        hoi_score = torch.zeros(1, 117)
        person_score = scores[0]
        object_score = scores[1]
        interaction_score = scores[2]
        action_scores = (pred_logits[0] + pred_logits[-1]) / 2

        hoi_scores = torch.zeros(1, 117)

        sub_bbox.append(torch.tensor(bbox[:4]))
        obj_bbox.append(torch.tensor(bbox[4:]))
        obj_label.append(int(HICO_OBJECTSname2index[object_name]))

        for hid, (potential, action_score) in enumerate(zip(potential_list, action_scores[0])):
            action_name, _ = sentence2action_object[potential]
            hoi_scores[0][int(HICO_ACTIONname2index[action_name])] = float(action_score ** 1.3) * (object_score ** 1.3) * (person_score ** 1.3) * (interaction_score ** 1.3)

        verb_score.append(hoi_scores)

    if len(verb_score) > 0:
        verb_score = torch.cat(verb_score, dim=0)
        # print(verb_score.shape)
        sub_bbox = torch.stack(sub_bbox)
        obj_bbox = torch.stack(obj_bbox)
        obj_label = torch.tensor(obj_label)

        sl = torch.full_like(obj_label, 0)
        l = torch.cat((sl, obj_label))
        b = torch.cat((sub_bbox, obj_bbox))
        results.append({'labels': l, 'boxes': b})

        ids = torch.arange(b.shape[0])
        results[-1].update({'verb_scores': verb_score.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})
    else:
        results.append({'labels': [], 'boxes': [], 'verb_scores': [], 'sub_ids': [], 'obj_ids': []})
        

def parse_prediction(predictions):
    pred_hoi_ids = []
    pred_person_bboxs = []
    pred_object_bboxs = []
    format_error = 0
    for prediction in predictions.split("\n")[:-1]:
        try:
            pred_hoi_id, pred_person_bbox, pred_object_bbox = extract_object_action_bbox(prediction)
            pred_hoi_ids.append(pred_hoi_id)
            pred_person_bboxs.append(pred_person_bbox)
            pred_object_bboxs.append(pred_object_bbox)
        except:
            print(prediction)
            format_error += 1
        
    return pred_hoi_ids, pred_person_bboxs, pred_object_bboxs, format_error


def calc_ap(pred, keys, pred_bboxes, gt_boxes):

    # if len(pred) == 0:
    #     return 0, 0

    hit = []
    npos = 0
    used = {}

    w_index = [0, 2]
    h_index = [1, 3]

    for key in gt_boxes.keys():
        npos += len(gt_boxes[key])
        used[key] = set()

    for i in range(min(len(pred), 100000)):
        key = keys[i]
        pred_hoi_id = pred[i]
        pred_bbox = pred_bboxes[i]
        if key in gt_boxes:
            k = -1
            maxi = 0.0
            for i in range(len(gt_boxes[key])):
                object_gt_bbox = gt_boxes[key][i][4]
                person_gt_bbox = gt_boxes[key][i][3]
                width = gt_boxes[key][i][1]
                height = gt_boxes[key][i][2]

                gt_bbox = np.array(person_gt_bbox + object_gt_bbox)
                pred_person_bbox = []
                pred_object_bbox = []

                for t in range(4):
                    if t in w_index:
                        pred_person_bbox.append(width * pred_bbox[0][t])
                        pred_object_bbox.append(width * pred_bbox[1][t])
                    elif t in h_index:
                        pred_person_bbox.append(height * pred_bbox[0][t])
                        pred_object_bbox.append(height * pred_bbox[1][t])
                tmp = calc_hit(np.array(pred_person_bbox + pred_object_bbox), gt_bbox)
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)

    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos if npos > 0 else hit / (npos + 1e-8)
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec) if len(rec) else 0


def calculate_precision_recall(args):
    data_list = json.load(open(args.question_file, "r"))

    gts = []

    for data_item in data_list:
        boxes = [obj['bbox'] for obj in data_item['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = [HICO_OBJECTSname2index[HICO_OBJECTSid2information[obj['category_id']]['name']] for obj in data_item['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)
        target = {}
        target['orig_size'] = torch.as_tensor([int(data_item['height']), int(data_item['width'])])
        target['size'] = torch.as_tensor([int(data_item['height']), int(data_item['width'])])

        scale_fct = torch.Tensor([float(data_item['width']), float(data_item['height']), float(data_item['width']), float(data_item['height'])]).unsqueeze(0) # 1,4 

        target['boxes'] = boxes / scale_fct
        target['labels'] = classes
        target['id'] = data_item['img_id']

        hois = []
        for hoi in data_item['hoi_annotation']:
            hois.append((hoi['subject_id'], hoi['object_id'], hoi['category_id'] - 1))
        target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
        gts.append(target)

    format_error = 0
    pred_results = []

    lines = []
    lines += torch.load(f"{args.answer_file}/answer0.pkl")
    lines += torch.load(f"{args.answer_file}/answer1.pkl")
    lines += torch.load(f"{args.answer_file}/answer2.pkl")
    lines += torch.load(f"{args.answer_file}/answer3.pkl")

    for index, line in enumerate(lines):
        parse_prediction_group(pred_results, line)

    assert len(pred_results) == len(gts)
    evaluator = HICOEvaluator(pred_results, gts, )
    stats = evaluator.evaluate()
    print(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="tables/answer.jsonl")
    parser.add_argument("--group", action='store_true')
    args = parser.parse_args()

    HICO_OBJECTSid2information = {item['id']: item for item in HICO_OBJECTS}
    HICO_OBJECTSname2index = {item['name']: index for index, item in enumerate(HICO_OBJECTS)}
    HICO_ACTIONname2index = {item['name']: item['id'] for item in HICO_ACTIONS}
    HICO_NAME2index = {(item['action'], item['object']): index for index, item in enumerate(HICO_INTERACTIONS)}

    calculate_precision_recall(args)
