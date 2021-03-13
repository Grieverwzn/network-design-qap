from scipy.optimize import linear_sum_assignment


def Hungarian_1(assignment_mat):
    # location_ind, building_ind = linear_sum_assignment(assignment_mat)
    building_ind, location_ind = linear_sum_assignment(assignment_mat)
    # value = assignment_mat[location_ind, building_ind].sum()
    value = assignment_mat[building_ind, location_ind].sum()
    return {'building_ind': building_ind, 'location_ind': location_ind, 'value': value}


def Hungarian_2(assignment_mat):
    location_ind, building_ind = linear_sum_assignment(assignment_mat)
    # building_ind, location_ind = linear_sum_assignment(assignment_mat)
    value = assignment_mat[location_ind, building_ind].sum()
    # value = assignment_mat[building_ind, location_ind].sum()
    return {'building_ind': building_ind, 'location_ind': location_ind, 'value': value}