#!/usr/bin/env python3
""" 
calculate the monetary value of correctly predicted product matches.

---> for each candidate pair, the representative pair price is:
        pair_price = mean(left_entity_price, right_entity_price)

the primary metric is true-positive price coverage: the sum of pair prices for
samples whose ground-truth label is 1 and whose predicted label is also 1.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRICE_KEYS = (
    "price",
    "left_price",
    "right_price",
    "price_left",
    "price_right",
    "ltable_price",
    "rtable_price",
)

TRUE_LABEL_KEYS = ("label", "gold", "gold_label", "target", "y", "ground_truth")
PRED_LABEL_KEYS = (
    "prediction",
    "predicted_label",
    "pred_label",
    "pred",
    "output",
    "match",
    "label",
)
PROBABILITY_KEYS = (
    "match_probability",
    "probability",
    "prob",
    "score",
    "confidence",
    "match_score",
)
ID_KEYS = ("id", "pair_id", "sample_id", "index", "idx")
DOMAIN_KEYS = ("domain", "price_domain", "category", "group")
LEFT_KEYS = ("left", "ltable", "record_left", "entity_left", "entity1", "record1")
RIGHT_KEYS = ("right", "rtable", "record_right", "entity_right", "entity2", "record2")

# captures a price attribute in DITTO serialization
# stops at the next [COL] marker rather than assuming that price is the final attribute
DITTO_PRICE_RE = re.compile(
    r"(?:\[COL\]|COL)\s*price\s*(?:\[VAL\]|VAL)\s*"
    r"(.*?)(?=\s*(?:\[COL\]|COL)\s+|$)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sum the average pair prices of correctly predicted matches "
            "(ground truth = 1 and prediction = 1)."
        )
    )
    parser.add_argument("--task", default="", help="Optional task name recorded in the report.")
    parser.add_argument("--input_path", required=True, help="Labeled candidate-pair JSONL or native DITTO TXT test file.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL or text file.")
    parser.add_argument("--result_path", required=True, help="Path for the text summary report.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used when predictions contain probabilities rather than labels (default: 0.5).")
    parser.add_argument("--strict", action="store_true", help="Fail on missing prices, labels, or prediction/input length mismatches.")
    parser.add_argument("--domain_low_max", type=float, default=None, help="Optional upper bound for the low-price domain when domain labels are absent.")
    parser.add_argument("--domain_medium_max", type=float, default=None, help="Optional upper bound for the medium-price domain when domain labels are absent.")
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_inputs(path: Path) -> List[Dict[str, Any]]:
    """
    load labeled candidate pairs from JSONL or native DITTO TXT.

    JSONL supports either objects or arrays:
        {"left": ..., "right": ..., "label": 1}
        [left, right, 1]

    native DITTO TXT is expected to contain:
        left_serialization<TAB>right_serialization<TAB>label
        ---> preference is to use this
    """
    records: List[Dict[str, Any]] = []

    # DITTO TXT format
    if path.suffix.lower() == ".txt":
        for line_no, line in enumerate(read_nonempty_lines(path), start=1):
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    "Expected DITTO TXT line as left<TAB>right<TAB>label on "
                    "line {} of {}; found {} fields.".format(
                        line_no, path, len(parts)
                    )
                )
            records.append({
                "left": parts[0],
                "right": parts[1],
                "label": parts[-1],
            })
        return records

    # JSONL format
    for line_no, line in enumerate(read_nonempty_lines(path), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON on line {} of {}: {}".format(line_no, path, exc)
            )

        if isinstance(value, dict):
            records.append(value)
            continue

        if isinstance(value, (list, tuple)):
            if len(value) < 2:
                raise ValueError(
                    "Expected at least [left, right] on line {} of {}; "
                    "found an array of length {}.".format(
                        line_no, path, len(value)
                    )
                )

            record: Dict[str, Any] = {
                "left": value[0],
                "right": value[1],
            }
            if len(value) >= 3:
                record["label"] = value[2]
            if len(value) >= 4:
                record["domain"] = value[3]

            records.append(record)
            continue

        raise ValueError(
            "Expected a JSON object or [left, right, label] array on line "
            "{} of {}; found {}.".format(
                line_no, path, type(value).__name__
            )
        )

    return records


def first_present(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    # find first relevant attribute
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_binary_label(value: Any, threshold: float = 0.5) -> Optional[int]:
    """normalize binary label formatting"""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number in (0.0, 1.0):
            return int(number)
        return int(number >= threshold)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "match", "matched", "positive"}:
        return 1
    if text in {"0", "false", "no", "non-match", "nonmatch", "unmatched", "negative"}:
        return 0
    try:
        return int(float(text) >= threshold)
    except ValueError:
        return None


def parse_money(value: Any) -> Optional[float]:
    """convert a price-like value to a finite non-negative float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "unknown"}:
            return None
        # handle common price ranges such as "$10 - $14" by averaging them.
        numbers = NUMBER_RE.findall(text)
        if not numbers:
            return None
        parsed = [float(token.replace(",", "")) for token in numbers]
        number = sum(parsed) / len(parsed)

    if not math.isfinite(number) or number < 0:
        return None
    return number


def price_from_serialized(text: Any) -> Optional[float]:
    # extract price from serialized text
    if not isinstance(text, str):
        return None
    match = DITTO_PRICE_RE.search(text)
    if not match:
        return None
    return parse_money(match.group(1))


def price_from_entity(entity: Any) -> Optional[float]:
    # extract price from single entity
    if entity is None:
        return None
    if isinstance(entity, dict):
        # exact key first, then case-insensitive lookup.
        if "price" in entity:
            result = parse_money(entity["price"])
            if result is not None:
                return result
        for key, value in entity.items():
            if str(key).strip().lower() == "price":
                result = parse_money(value)
                if result is not None:
                    return result
        # some formats wrap attributes one level deeper
        for value in entity.values():
            if isinstance(value, dict):
                result = price_from_entity(value)
                if result is not None:
                    return result
        return None
    if isinstance(entity, str):
        return price_from_serialized(entity)
    return None


def extract_pair_prices(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """extract prices for a given pair from entity collection"""
    # --> use average of (left, right) prices, or whichever one is present if only one is missing
    left = first_present(record, LEFT_KEYS)
    right = first_present(record, RIGHT_KEYS)
    left_price = price_from_entity(left)
    right_price = price_from_entity(right)

    # explicit top-level left/right fields override only when the entity price
    # could not be found.
    if left_price is None:
        left_price = parse_money(first_present(record, ("left_price", "price_left", "ltable_price")))
    if right_price is None:
        right_price = parse_money(first_present(record, ("right_price", "price_right", "rtable_price")))

    # common matcher input schema: {"text_left": ..., "text_right": ...}
    if left_price is None:
        left_price = price_from_serialized(first_present(record, ("text_left", "left_text", "sentence1")))
    if right_price is None:
        right_price = price_from_serialized(first_present(record, ("text_right", "right_text", "sentence2")))

    # DITTO pair may be stored in one field separated by tabs or [SEP].
    if left_price is None or right_price is None:
        combined = first_present(record, ("text", "pair", "input", "serialized"))
        if isinstance(combined, str):
            parts = re.split(r"\t|\[SEP\]", combined, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                if left_price is None:
                    left_price = price_from_serialized(parts[0])
                if right_price is None:
                    right_price = price_from_serialized(parts[1])

    return left_price, right_price


def representative_pair_price(record: Dict[str, Any]) -> Optional[float]:
    left_price, right_price = extract_pair_prices(record)

    # average the prices when both entities have one.
    if left_price is not None and right_price is not None:
        return (left_price + right_price) / 2.0

    # when only one entity has a price, use that price directly.
    if left_price is not None:
        return left_price

    if right_price is not None:
        return right_price

    # fall back to a precomputed pair-level price.
    return parse_money(
        first_present(
            record,
            ("pair_price", "average_price", "avg_price", "price")
        )
    )


def extract_true_label(record: Dict[str, Any], threshold: float) -> Optional[int]:
    # extract ground truth matching label
    return normalize_binary_label(first_present(record, TRUE_LABEL_KEYS), threshold)


def extract_prediction(record: Dict[str, Any], threshold: float) -> Optional[int]:
    # extract the model prediction from result files
    value = first_present(record, PRED_LABEL_KEYS)
    label = normalize_binary_label(value, threshold)
    if label is not None:
        return label
    probability = first_present(record, PROBABILITY_KEYS)
    return normalize_binary_label(probability, threshold)


def load_predictions(path: Path, threshold: float) -> List[Dict[str, Any]]:
    # load the matching predictions from a model EM run
    lines = read_nonempty_lines(path)
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # also support a plain file with one 0/1 label or probability per line.
            label = normalize_binary_label(line, threshold)
            if label is None:
                raise ValueError(
                    "Could not parse prediction on line {} of {}: {!r}".format(line_no, path, line)
                )
            records.append({"prediction": label})
            continue

        if isinstance(value, dict):
            records.append(value)
        else:
            records.append({"prediction": value})
    return records


def align_records(inputs: List[Dict[str, Any]], predictions: List[Dict[str, Any]], strict: bool) -> Iterable[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """align by a shared identifier when possible; otherwise align by line."""
    pred_by_id: Dict[str, Dict[str, Any]] = {}
    usable_ids = True
    for pred in predictions:
        pred_id = first_present(pred, ID_KEYS)
        if pred_id is None:
            usable_ids = False
            break
        pred_by_id[str(pred_id)] = pred

    input_has_ids = all(first_present(item, ID_KEYS) is not None for item in inputs)
    if usable_ids and input_has_ids:
        for index, item in enumerate(inputs):
            item_id = str(first_present(item, ID_KEYS))
            if item_id not in pred_by_id:
                if strict:
                    raise ValueError("No prediction found for input id {!r}".format(item_id))
                continue
            yield index, item, pred_by_id[item_id]
        return

    if len(inputs) != len(predictions):
        message = "Input/prediction length mismatch: {} inputs versus {} predictions.".format(
            len(inputs), len(predictions)
        )
        if strict:
            raise ValueError(message)
        print("WARNING: " + message + " Using the first {} aligned rows.".format(min(len(inputs), len(predictions))), file=sys.stderr)

    for index, (item, pred) in enumerate(zip(inputs, predictions)):
        yield index, item, pred


def infer_domain(record: Dict[str, Any], pair_price: float, low_max: Optional[float], medium_max: Optional[float]) -> str:
    # infer domain of record based on defined price ranges for (low, medium, high)
    existing = first_present(record, DOMAIN_KEYS)
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    if low_max is not None and medium_max is not None:
        if pair_price <= low_max:
            return "low"
        if pair_price <= medium_max:
            return "medium"
        return "high"
    return "all"


def money(value: float) -> str:
    return "${:,.2f}".format(value)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    prediction_path = Path(args.predictions)
    result_path = Path(args.result_path)

    # load both the labeled pairs and model predictions
    inputs = load_inputs(input_path)
    predictions = load_predictions(prediction_path, args.threshold)

    # extract predictions and labels from loaded data
    counts = defaultdict(int)
    values = defaultdict(float)
    domain_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    skipped_examples: List[str] = []
    for index, item, prediction_record in align_records(inputs, predictions, args.strict):
        true_label = extract_true_label(item, args.threshold)
        if true_label is None:
            # some matcher outputs preserve the gold label.
            true_label = extract_true_label(prediction_record, args.threshold)
        predicted_label = extract_prediction(prediction_record, args.threshold)
        pair_price = representative_pair_price(item)

        # count missing rows (e.g. missing label, predictions, price missing for both entites)
        missing = []
        if true_label is None:
            missing.append("ground-truth label")
        if predicted_label is None:
            missing.append("prediction")
        if pair_price is None:
            missing.append("price")
        if missing:
            counts["skipped"] += 1
            description = "row {} missing {}".format(index + 1, ", ".join(missing))
            if len(skipped_examples) < 10:
                skipped_examples.append(description)
            if args.strict:
                raise ValueError(description)
            continue

        # determine domain and update counts
        domain = infer_domain(item, pair_price, args.domain_low_max, args.domain_medium_max)
        counts["evaluated"] += 1
        values["all_pairs"] += pair_price
        domain_stats[domain]["evaluated"] += 1
        domain_stats[domain]["all_value"] += pair_price

        # track true/false negatives/positives
        if true_label == 1:
            counts["actual_matches"] += 1
            values["actual_matches"] += pair_price
            domain_stats[domain]["actual_matches"] += 1
            domain_stats[domain]["actual_match_value"] += pair_price
        if predicted_label == 1:
            counts["predicted_matches"] += 1
            values["predicted_matches"] += pair_price
        if true_label == 1 and predicted_label == 1:
            counts["true_positives"] += 1
            values["true_positives"] += pair_price
            domain_stats[domain]["true_positives"] += 1
            domain_stats[domain]["true_positive_value"] += pair_price
        elif true_label == 0 and predicted_label == 1:
            counts["false_positives"] += 1
            values["false_positives"] += pair_price
            domain_stats[domain]["false_positives"] += 1
            domain_stats[domain]["false_positive_value"] += pair_price
        elif true_label == 1 and predicted_label == 0:
            counts["false_negatives"] += 1
            values["false_negatives"] += pair_price
            domain_stats[domain]["false_negatives"] += 1
            domain_stats[domain]["false_negative_value"] += pair_price
        else:
            counts["true_negatives"] += 1

    # compute recall price value
    actual_value = values["actual_matches"]
    covered_value = values["true_positives"]
    value_recall = covered_value / actual_value if actual_value else 0.0
    ordinary_recall = (counts["true_positives"] / counts["actual_matches"] if counts["actual_matches"] else 0.0)

    # format calculated information
    lines = []
    lines.append("Price-Based Entity Matching Case Study")
    lines.append("=" * 39)
    if args.task:
        lines.append("Task: {}".format(args.task))
    lines.append("Input: {}".format(input_path))
    lines.append("Predictions: {}".format(prediction_path))
    lines.append("Prediction threshold: {:.4f}".format(args.threshold))
    lines.append("")
    lines.append("Primary result")
    lines.append("--------------")
    lines.append("Correct matching predictions (true positives): {:,}".format(counts["true_positives"]))
    lines.append("Total price of correct matching predictions: {}".format(money(covered_value)))
    lines.append("Price-weighted match recall: {:.2%}".format(value_recall))
    lines.append("")
    lines.append("Overall breakdown")
    lines.append("-----------------")
    lines.append("Evaluated pairs: {:,}".format(counts["evaluated"]))
    lines.append("Skipped pairs: {:,}".format(counts["skipped"]))
    lines.append("Ground-truth matches: {:,}".format(counts["actual_matches"]))
    lines.append("Predicted matches: {:,}".format(counts["predicted_matches"]))
    lines.append("True positives: {:,}".format(counts["true_positives"]))
    lines.append("False positives: {:,}".format(counts["false_positives"]))
    lines.append("False negatives: {:,}".format(counts["false_negatives"]))
    lines.append("True negatives: {:,}".format(counts["true_negatives"]))
    lines.append("Standard match recall: {:.2%}".format(ordinary_recall))
    lines.append("Total value of all ground-truth matches: {}".format(money(actual_value)))
    lines.append("Recovered true-match value: {}".format(money(covered_value)))
    lines.append("Missed true-match value: {}".format(money(values["false_negatives"])))
    lines.append("False-positive predicted value: {}".format(money(values["false_positives"])))

    # report domain-level information 
    lines.append("")
    lines.append("Domain breakdown")
    lines.append("----------------")
    for domain in sorted(domain_stats):
        stats = domain_stats[domain]
        domain_actual_value = stats["actual_match_value"]
        domain_covered_value = stats["true_positive_value"]
        domain_value_recall = domain_covered_value / domain_actual_value if domain_actual_value else 0.0
        lines.append("{}:".format(domain))
        lines.append("  Evaluated pairs: {:,}".format(int(stats["evaluated"])))
        lines.append("  Ground-truth matches: {:,}".format(int(stats["actual_matches"])))
        lines.append("  Correct matching predictions: {:,}".format(int(stats["true_positives"])))
        lines.append("  Correct-match price total: {}".format(money(domain_covered_value)))
        lines.append("  Ground-truth match price total: {}".format(money(domain_actual_value)))
        lines.append("  Missed-match price total: {}".format(money(stats["false_negative_value"])))
        lines.append("  Price-weighted match recall: {:.2%}".format(domain_value_recall))

    if skipped_examples:
        lines.append("")
        lines.append("Skipped-row examples")
        lines.append("--------------------")
        lines.extend("- " + item for item in skipped_examples)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    result_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print("Wrote report to: {}".format(result_path))


if __name__ == "__main__":
    main()
