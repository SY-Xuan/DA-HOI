from transformers import GroundingDinoProcessor
from transformers.image_transforms import center_to_corners_format
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def get_phrases_from_posmap(posmaps, input_ids):
    """Get token ids of phrases from posmaps and input_ids.

    Args:
        posmaps (`torch.BoolTensor` of shape `(num_boxes, hidden_size)`):
            A boolean tensor of text-thresholded logits related to the detected bounding boxes.
        input_ids (`torch.LongTensor`) of shape `(sequence_length, )`):
            A tensor of token ids.
    """
    left_idx = 0
    right_idx = posmaps.shape[-1] - 1

    # Avoiding altering the input tensor
    posmaps = posmaps.clone()

    posmaps[:, 0 : left_idx + 1] = False
    posmaps[:, right_idx:] = False

    token_ids = []
    for posmap in posmaps:
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids.append([input_ids[i] for i in non_zero_idx])

    return token_ids

SPECIAL_TOKENS = [101, 102, 1012, 1029]

def build_label_maps(logits=None, input_ids: torch.LongTensor=None) -> Tuple[torch.FloatTensor]:
    """
    Computes a mapping between tokens and their corresponding labels, where `num_labels` is determined by the number of classes in the input prompt.
    The function identifies segments of tokens between specific delimiter tokens and generates label maps for those segments.
    Args:
        logits (`torch.Tensor` of shape `(batch_size, seq_length, hidden_size)`):
            The output logits from the model, where `hidden_size` corresponds to the dimension of the model's output features.

        input_ids (`torch.Tensor` of shape `(batch_size, seq_length)`):
            The input token IDs corresponding to the input prompt. For example, given the prompt "fish. shark.",
            `input_ids` might look like `[101, 3869, 1012, 11420, 1012, 102]` where each number corresponds to a token including special tokens.
    Returns:
        tuple: A tuple containing label maps for each instance in the batch.
        - label_maps (tuple of `torch.Tensor`):
            A tuple of tensors, where each tensor in the tuple corresponds to an instance in the batch. Each tensor
            has shape `(num_labels, hidden_size)` and contains binary values (0 or 1), where `1` indicates the tokens
            that are associated with a specific label (class) between delimiter tokens, and `0` elsewhere.
    Example:
        Given an input prompt "fish. shark." and corresponding `input_ids` as `[101, 3869, 1012, 11420, 1012, 102]`:
        - The function identifies the tokens for "fish" (IDs `[3869]`) and "shark" (IDs `[11420]`).
        - The function then constructs label maps for these tokens, where each label map indicates which tokens
          correspond to which label between the delimiter tokens (e.g., between the period `.`).
        - The output is a tuple of label maps, one for each instance in the batch.
    Note:
        - `SPECIAL_TOKENS` should be a predefined list of tokens that are considered special (e.g., `[CLS]`, `[SEP]`, etc.).
    """
    if logits is not None:
        max_seq_len = logits.shape[-1]
    else:
        max_seq_len = input_ids.shape[-1]
    # Add [PAD] token to the list of special tokens
    delimiter_tokens = torch.tensor(SPECIAL_TOKENS + [0], device=input_ids.device)

    delimiter_token_masks = torch.isin(input_ids, delimiter_tokens)
    label_groups = torch.cumsum(delimiter_token_masks, dim=1) * (~delimiter_token_masks).to(torch.int32)

    label_groups_with_zero_max = label_groups + (delimiter_token_masks) * 10000
    label_groups_with_zero_max = F.pad(label_groups_with_zero_max, (0, max_seq_len - label_groups_with_zero_max.shape[1]), value=10000)

    label_maps = ()

    # Iterate over batch dimension as we can have different number of labels
    for label_group in label_groups:
        # `label_group` is a tensor of shape `(seq_len,)` with zeros for non-label tokens and integers for label tokens
        # label tokens with same integer value are part of the same label group

        # Get unique labels and exclude 0 (i.e. non-label tokens)
        unique_labels = torch.unique(label_group)[1:, None]
        num_labels = unique_labels.shape[0]

        # Create one-hot encoding for each label group
        label_map = label_group.unsqueeze(0).repeat(num_labels, 1)
        label_map = torch.where(label_map == unique_labels, 1, 0)

        # Pad label_map to match `max_seq_len`
        label_map = F.pad(label_map, (0, max_seq_len - label_map.shape[1]), value=0)

        label_maps += (label_map,)

    return label_maps, label_groups_with_zero_max


def _is_list_of_candidate_labels(text) -> bool:
    """Check that text is list/tuple of strings and each string is a candidate label and not merged candidate labels text.
    Merged candidate labels text is a string with candidate labels separated by a dot.
    """
    if isinstance(text, (list, tuple)):
        return all(isinstance(t, str) and "." not in t for t in text)
    return False


def _merge_candidate_labels_text(text) -> str:
    """
    Merge candidate labels text into a single string. Ensure all labels are lowercase.
    For example, ["A cat", "a dog"] -> "a cat. a dog."
    """
    labels = [t.strip().lower() for t in text]  # ensure lowercase
    merged_labels_str = ". ".join(labels) + "."  # join with dot and add a dot at the end
    return merged_labels_str


class NewGroundingDinoProcessor(GroundingDinoProcessor):
    def post_process_grounded_object_detection_filter(
        self,
        outputs,
        input_ids,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes = None,
        topk = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        label_maps, label_groups_with_zero_max = build_label_maps(logits=logits, input_ids=input_ids)

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        probs = probs * (label_groups_with_zero_max != 10000).float()
        scores, max_index = torch.max(probs, dim=-1)  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        if topk is not None:
            topk_scores = torch.topk(scores, k=topk+1, dim=-1)[0]
            box_threshold = topk_scores[:, -1]

        results = []
        for idx, (s, b, p, max_i, label_group) in enumerate(zip(scores, boxes, probs, max_index, label_groups_with_zero_max)):
            score = s[s > box_threshold]
            box = b[s > box_threshold]
            prob = p[s > box_threshold]
            # NOTE: double filter ?
            # NOTE: remember to deal with zero detection
            group = label_group[max_i[s > box_threshold]] - 1 # num, cls_id
            label_maps_id = label_maps[idx][group].bool()[:, :input_ids[idx].shape[-1]]

            # label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            label = []
            for maps_id in label_maps_id:
                label.append(self.batch_decode([input_ids[idx][maps_id]]))
            # label = self.batch_decode(input_ids[idx].unsqueeze(0).repeat(label_maps_id.shape[0], 1)[label_maps_id])
            results.append({"scores": score, "labels": label, "boxes": box, "labels_id": group})

        return results

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
            score = s[s > box_threshold]
            box = b[s > box_threshold]
            prob = p[s > box_threshold]
            label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            label = self.batch_decode(label_ids)
            results.append({"scores": score, "labels": label, "boxes": box})

        return results
