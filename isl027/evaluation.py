"""
Evaluation script for the Resume Optimization System.

This script runs a series of functional tests to evaluate the core components of the application:
1. Skill extraction evaluation using manually labelled examples.
2. Candidate skill-match checks against expected results.
3. Fake-job detector checks using rule-based test examples.
4. Precision, Recall, and F1-score calculations for the evaluation metrics.
5. CSV export of the detailed evaluation results for the final report.

Note:
These are small, manually prepared functional tests designed for academic evaluation. 
They are not a large-scale real-world benchmark.
"""

import csv
from model_utils import analyze_match, extract_skills
from nlp_features import detect_fake_job


def calculate_metrics(true_positive, false_positive, false_negative):
    """
    Calculates Precision, Recall, and F1-Score for the evaluation tests.
    
    Parameters:
    - true_positive: Number of correctly predicted positive cases.
    - false_positive: Number of incorrectly predicted positive cases.
    - false_negative: Number of incorrectly predicted negative cases.
    
    Returns:
    A dictionary containing the rounded precision, recall, and f1 scores.
    """
    
    # Precision: How many of the predicted positives were actually correct
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    
    # Recall: How many of the actual positives were successfully predicted
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    
    # F1-score: The harmonic mean of precision and recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def evaluate_skill_extraction():
    """
    Evaluates how accurately the regex-based skill extractor finds skills.
    It compares the predicted skills against the manually specified 'gold' labels.
    """
    print("\n" + "=" * 75)
    print("SKILL EXTRACTION EVALUATION")
    print("=" * 75)

    # Gold labels are manually specified for these test examples.
    # These represent the ideal expected output for each input text.
    test_cases = [
        {
            "name": "Software Developer",
            "text": "Python, SQL, Django, AWS and Machine Learning",
            "gold": {"python", "sql", "django", "aws", "machine learning"}
        },
        {
            "name": "Frontend Developer",
            "text": "HTML, CSS, React, JavaScript and basic Python",
            "gold": {"html", "css", "react", "javascript", "python"}
        },
        {
            "name": "Data Analyst",
            "text": "Python, pandas, numpy, Excel and Power BI",
            "gold": {"python", "pandas", "numpy", "excel", "power bi"}
        },
        {
            "name": "Unrelated Profile",
            "text": "Laundry supervisor with fabric sorting and hygiene experience",
            "gold": set()
        },
        {
            "name": "Cloud Developer",
            "text": "Docker, Kubernetes, Linux, AWS and Git",
            "gold": {"docker", "kubernetes", "linux", "aws", "git"}
        }
    ]

    # Initialize variables to track true positives, false positives, and false negatives
    tp = fp = fn = 0
    rows = []

    # Loop through all test cases to calculate extracted skills and compare them
    for case in test_cases:
        # Extract skills using the actual function from model_utils
        predicted = set(extract_skills(case["text"]))
        gold = case["gold"]

        # Calculate the intersections and differences for this specific case
        case_tp = len(predicted & gold)
        case_fp = len(predicted - gold)
        case_fn = len(gold - predicted)

        # Accumulate the overall counts
        tp += case_tp
        fp += case_fp
        fn += case_fn

        # Prepare the row data for CSV export
        rows.append({
            "Test Type": "Skill Extraction",
            "Test Name": case["name"],
            "Expected": ", ".join(sorted(gold)) or "None",
            "Predicted": ", ".join(sorted(predicted)) or "None",
            "Score": "",
            "Level": "",
            "Result": "PASS" if predicted == gold else "CHECK"
        })

        # Print the individual test result to the console
        print(f"\n{case['name']}")
        print(f"Expected:  {sorted(gold)}")
        print(f"Predicted: {sorted(predicted)}")
        print("Result:", "PASS" if predicted == gold else "CHECK")

    # Calculate the overall performance metrics for skill extraction
    metrics = calculate_metrics(tp, fp, fn)

    print("\nSkill Extraction Metrics")
    print("-" * 45)
    print(f"True Positives : {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1']:.4f}")

    return metrics, rows


def evaluate_candidate_matching():
    """
    Tests if the CV-JD matcher correctly identifies matched and missing skills.
    This ensures the core matching logic is working as intended.
    """
    print("\n" + "=" * 75)
    print("CV AND JOB DESCRIPTION MATCHING TEST")
    print("=" * 75)

    # Reference job description used for all candidates
    job_description = (
        "We need a Python Developer with expertise in "
        "Python, SQL, Django, AWS, and Machine Learning."
    )

    # Define three test candidates with expected matched and missing skills
    cases = [
        {
            "name": "Candidate A - Strong Skill Match",
            "cv": (
                "Python Developer with Python, SQL, Django, AWS "
                "and Machine Learning experience."
            ),
            "expected_matched": {
                "python", "sql", "django", "aws", "machine learning"
            },
            "expected_missing": set()
        },
        {
            "name": "Candidate B - Partial Skill Match",
            "cv": (
                "Frontend Developer with HTML, CSS, React, "
                "JavaScript and basic Python knowledge."
            ),
            "expected_matched": {"python"},
            "expected_missing": {"sql", "django", "aws", "machine learning"}
        },
        {
            "name": "Candidate C - Low Skill Match",
            "cv": (
                "Laundry Supervisor with experience in fabric "
                "sorting, ironing and hygiene management."
            ),
            "expected_matched": set(),
            "expected_missing": {
                "python", "sql", "django", "aws", "machine learning"
            }
        }
    ]

    rows = []

    # Run the matching algorithm for each test candidate
    for case in cases:
        # Analyze the candidate's CV against the reference JD
        result = analyze_match(case["cv"], job_description)

        # Convert the output lists to sets for easier comparison
        actual_matched = set(result["matched"])
        actual_missing = set(result["missing"])

        # Check if the actual output matches the expected output
        matched_ok = actual_matched == case["expected_matched"]
        missing_ok = actual_missing == case["expected_missing"]
        test_status = "PASS" if matched_ok and missing_ok else "CHECK"

        # Print the detailed results to the console
        print(f"\n{case['name']}")
        print(f"Match Score: {result['score']}%")
        print(f"Match Level: {result['level']}")
        print(f"Matched Skills: {sorted(actual_matched)}")
        print(f"Missing Skills: {sorted(actual_missing)}")
        print(f"Expected Skill Check: {test_status}")

        # Prepare the row data for CSV export
        rows.append({
            "Test Type": "Candidate Matching",
            "Test Name": case["name"],
            "Expected": ", ".join(sorted(case["expected_matched"])) or "None",
            "Predicted": ", ".join(sorted(actual_matched)) or "None",
            "Score": result["score"],
            "Level": result["level"],
            "Result": test_status
        })

    return rows


def evaluate_fake_job_detector():
    """
    Evaluates the rule-based scam detector using both normal and fake JDs.
    This tests the reliability of the heuristic based detection.
    """
    print("\n" + "=" * 75)
    print("FAKE JOB DETECTOR EVALUATION")
    print("=" * 75)

    # A mix of scam indicators and normal job postings to test the detector
    test_cases = [
        {
            "name": "Registration Fee Request",
            "text": "Pay a registration fee before joining this job.",
            "expected": 1
        },
        {
            "name": "Suspicious Payment Method",
            "text": "Send the payment through Western Union to get hired.",
            "expected": 1
        },
        {
            "name": "Crypto Job Offer",
            "text": "Pay a deposit in Bitcoin to receive your job offer.",
            "expected": 1
        },
        {
            "name": "Suspicious Messaging Recruitment",
            "text": "Contact our recruiter on WhatsApp for an easy money offer.",
            "expected": 1
        },
        {
            "name": "Normal Company Application",
            "text": "Apply through our official company careers page. "
                    "The role requires Python and SQL experience.",
            "expected": 0
        },
        {
            "name": "Normal Technical Job",
            "text": "We are hiring a software developer. "
                    "Please submit your CV and attend a technical interview.",
            "expected": 0
        },
        {
            "name": "Normal Internship",
            "text": "The internship includes training, project work "
                    "and an interview with the engineering team.",
            "expected": 0
        },
        {
            "name": "Normal Data Analyst Vacancy",
            "text": "Candidates should have experience with Excel, SQL "
                    "and data analysis. Apply on the company website.",
            "expected": 0
        }
    ]

    # Initialize confusion matrix variables
    tp = fp = fn = tn = 0
    rows = []

    # Iterate through the cases and track the confusion matrix metrics
    for case in test_cases:
        output = detect_fake_job(case["text"])
        
        # Determine if the model predicted a scam based on the output string
        predicted = 1 if "Potential Red Flags Detected" in output else 0
        expected = case["expected"]

        # Update confusion matrix counts based on prediction vs actual
        if expected == 1 and predicted == 1:
            tp += 1
        elif expected == 0 and predicted == 1:
            fp += 1
        elif expected == 1 and predicted == 0:
            fn += 1
        else:
            tn += 1

        status = "PASS" if predicted == expected else "CHECK"

        # Print detailed results
        print(f"\n{case['name']}")
        print(f"Expected: {'Suspicious' if expected else 'Normal'}")
        print(f"Predicted: {'Suspicious' if predicted else 'No red flags found'}")
        print(f"Result: {status}")

        # Prepare row for CSV
        rows.append({
            "Test Type": "Fake Job Detector",
            "Test Name": case["name"],
            "Expected": expected,
            "Predicted": predicted,
            "Score": "",
            "Level": "",
            "Result": status
        })

    # Calculate precision, recall, and f1-score
    metrics = calculate_metrics(tp, fp, fn)

    # Calculate overall accuracy including true negatives
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    print("\nFake Job Detector Metrics")
    print("-" * 45)
    print(f"True Positives : {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives : {tn}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1']:.4f}")

    return metrics, rows


def save_results(rows):
    """
    Saves the evaluation results to a CSV file for reporting.
    This allows the user to include a detailed test log in their final report.
    """
    filename = "evaluation_results.csv"
    fieldnames = [
        "Test Type", "Test Name", "Expected",
        "Predicted", "Score", "Level", "Result"
    ]

    # Open the file and write the headers and rows
    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results saved to: {filename}")


def run_evaluation():
    """
    Main function to run all evaluation tests and print the final summary.
    This provides a complete overview of the system's performance.
    """
    print("\nRESUME OPTIMIZATION SYSTEM - MODEL EVALUATION")

    # Execute all three evaluation phases
    skill_metrics, skill_rows = evaluate_skill_extraction()
    candidate_rows = evaluate_candidate_matching()
    fake_metrics, fake_rows = evaluate_fake_job_detector()

    # Combine all test results and save them to a CSV file
    all_rows = skill_rows + candidate_rows + fake_rows
    save_results(all_rows)

    # Display the final summary of the metrics
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(
        "Skill Extraction - "
        f"Precision: {skill_metrics['precision']:.4f}, "
        f"Recall: {skill_metrics['recall']:.4f}, "
        f"F1: {skill_metrics['f1']:.4f}"
    )
    print(
        "Fake Job Detector - "
        f"Precision: {fake_metrics['precision']:.4f}, "
        f"Recall: {fake_metrics['recall']:.4f}, "
        f"F1: {fake_metrics['f1']:.4f}"
    )
    print("\nEvaluation completed.")
    print(
        "These results describe only the included test examples. "
        "They are not a measure of performance on all real-world CVs or job posts."
    )


if __name__ == "__main__":
    run_evaluation()
