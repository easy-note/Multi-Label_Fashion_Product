
# custom metric function for multi-head multi-category classification
def label_based_eval(outputs_dict, targets_dict):
    result = []
    for label in ['gender', 'articleType', 'season', 'usage']:
        acc = 0
        for (o, t) in zip(outputs_dict[label], targets_dict[label]):
            if o == t:
                acc += 1
        
        print(f"{label} accuracy : {acc/len(outputs_dict[label])}")
        