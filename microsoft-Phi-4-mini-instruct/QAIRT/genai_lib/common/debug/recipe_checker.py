import json
import csv
import argparse
csv_writer = None
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_dicts(dict1, dict2, context):
    global csv_writer
    '''
    the compare_dict function compares two dictionaries dict1 and dict2 recursively
    a. it iterates over the keys in dict1, if the  key is in dict2, it compares the corresponding values
    b. if the key is not present in dict2, we report it
    c. we also iterate over the keys in dict2 and check for their presence in dict1 and report if not present.
    params:
    1. dict1: dictionary representing the payload of a given PTQ technique/API
    2. dict2: dictionary representing the payload of a given PTQ technique/API
    3. context: A string describing the hierarchical path to the point of difference between the payloads
    '''
    for key in dict1:
        if key not in dict2:
            csv_writer.writerow([f"{context}: {key}", dict1[key], ""])
        else:
            compare_values(dict1[key], dict2[key], f"{context} -> {key}")

    for key in dict2:
        if key not in dict1:
            csv_writer.writerow([f"{context}: {key}", "", dict2[key]])

def compare_values(val1, val2, context):
    '''
    compares two values val1, and val2. it checks for the types of val1 and val2 and if it is a dictionary it
    invokes the compare_dict function recursively, else we compare the values if they match or not.
        params:
    1. val1: The value of a specific property in the PTQ technique/API payload being compared.
    2. val2: value of the same property in the PTQ technique/API payload being compared
    3. context: A string describing the hierarchical path to the point of difference between the payloads
    '''
    global csv_writer
    if isinstance(val1, dict) and isinstance(val2, dict):
        compare_dicts(val1, val2, context)
    elif val1 != val2:
        csv_writer.writerow([f"{context}", val1, val2])

def compare_json(json1, json2):
    '''
    This function is primarily responsible for comparing the PTQ techniques/ AIMET calls as captured in the recipe dumps.
    As the strict checking, we ensure that the two dumps contain the same number of techniques else, we report to user and stop.
    Next, we check if the order of techniques (determined by the Module_name+Operation name is aligned) is same,
    if not, we do a second check to ensure that the set of techniques is same across the two dumps for comparison.
    Once we have verified that either the order or the set matches, we then iterate over each function call captured and
    compare the payload within the Parameters of algo and the Additional Properties.
    params:
    1. json1: the JSON corresponding to the recipe dump1
    2. json2: the JSON corresponding to the recipe dump2
    '''
    global csv_writer
    report = []
    file = open('mismatches.csv', mode='w', newline='')
    csv_writer = csv.writer(file)
    # Write the header
    csv_writer.writerow(['Property', 'Value in API1', 'Value in API2'])

    api1 = json1.get('Quantization_PTQ_API', [])
    api2 = json2.get('Quantization_PTQ_API', [])

    environment_1 = json1.get('Environment', [])[0]
    environment_2 = json2.get('Environment', [])[0]
    compare_dicts(environment_1, environment_2,
                  "Environment")
    # Order and Element Matching
    ordered = True
    if len(api1) != len(api2):
        report.append("Number of PTQ techniques applied do not match.")
        return report
    else:
        for i, (dict1, dict2) in enumerate(zip(api1, api2)):
            if dict1.get('Module_name') != dict2.get('Module_name') and dict1.get('Operation_name') != dict2.get('Operation_name'):
                ordered=False
                break

        if not ordered:
            report.append(
                f"Warning: the order of PTQ is different between the two dumps. Proceeding with verify if the set of PTQ is same or not..")
            for dict1 in api1:
                dict2 = next((d for d in api2 if
                              d['Module_name'] == dict1['Module_name'] and d['Operation_name'] == dict1['Operation_name']),
                             None)
                if dict2 is None:
                    report.append(
                        f"PTQs do not match across the two dumps, please verify if the set of techniques applied matches.")
                    return report
            report.append(
                f"PTQs order does not match across the two dumps, but the set of techniques applied matches.")

    for dict1 in api1:
        dict2 = next((d for d in api2 if d['Module_name'] == dict1['Module_name'] and d['Operation_name'] == dict1['Operation_name']), None)
        # Compare Parameters_of_algo and Additional_properties
        compare_dicts(dict1.get('Parameters_of_algo', {}), dict2.get('Parameters_of_algo', {}),
                      f"Parameters_of_algo in {str(dict1.get('Operation_name'))}")
        compare_dicts(dict1.get('Additional_properties', {}), dict2.get('Additional_properties', {}),
                      f"Additional_properties in  {str(dict1.get('Operation_name'))}")
    file.close()
    report.append("\nNote: Please refer to the mismatches.csv file for details about the mismatches.")
    return report

def generate_report(report):
    '''
    This function reads in the report and writes it to the comparison_report.txt file.
    '''
    with open('comparison_report.txt', 'w') as file:
        for line in report:
            file.write(line + '\n')

if __name__ == "__main__":
    '''
    The script compares the JSON dumps generated from the recipe_logger.
    '''
    parser = argparse.ArgumentParser(description='''
                                                This script, when executed with two recipe dumps in JSON format, compares the environment and the Quantization PTQ API payload information between the two dumps. 
                                                It generates two files: `mismatches.csv`, which lists the mismatches along with the corresponding API, 
                                                and `report.txt`, which provides additional details on whether there is a discrepancy in the number of PTQ techniques applied between the two dumps.
                                                Usage:
                                                python recipe_checker.py path_to_recipe_dump1 path_to_recipe_dump2
                                                ''')
    parser.add_argument('json1_path', type=str, help='Path to the first JSON file')
    parser.add_argument('json2_path', type=str, help='Path to the second JSON file')

    args = parser.parse_args()

    json1 = load_json(args.json1_path)
    json2 = load_json(args.json2_path)

    report = compare_json(json1, json2)
    generate_report(report)
