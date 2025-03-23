import csv

def load_names_from_csv(filename):
    """
    Loads names from a CSV file that contains one column labeled "Name Words".
    """
    names = []
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            names.append(row["Name Words"].strip())
    return names

def compare_names(ground_truth, parser_output):
    """
    Compares the ground truth names with the parser's output names and computes
    precision, recall, and F1 score.
    """
    gt_set = set(ground_truth)
    parser_set = set(parser_output)
    
    true_positives = gt_set & parser_set
    false_positives = parser_set - gt_set
    false_negatives = gt_set - parser_set
    
    precision = len(true_positives) / len(parser_set) if parser_set else 0
    recall = len(true_positives) / len(gt_set) if gt_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, true_positives, false_positives, false_negatives

if __name__ == "__main__":
    # Filenames for the ground truth and parser output CSVs
    ground_truth_csv = "ground_truth.csv"
    parser_output_csv = "parser_output.csv"
    
    # Load the names from each CSV file (change parser_output_csv accordingly)
    ground_truth_names = load_names_from_csv(ground_truth_csv)
    parser_names = load_names_from_csv(parser_output_csv)
    
    # Compare the ground truth to the parser's output
    precision, recall, f1, true_positives, false_positives, false_negatives = compare_names(ground_truth_names, parser_names)
    
    # Print the evaluation metrics
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1 Score: {:.2f}".format(f1))
    print("True Positives:", true_positives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)
