# ©, 2024, Sirris
# owner: FFNG


def merge_lists_with_one_element(list_of_lists):
    merged_list = []
    output_list = []
    for l in list_of_lists:
        if len(l) == 1:
            merged_list.append(l[0])
        else:
            output_list.append(l)
    if len(merged_list) >= 1:
        output_list.append(merged_list)
    return output_list
            
    