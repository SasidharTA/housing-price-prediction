def _process_formula(formula, columns=[]):
    if formula.split()[-1].strip() == ".":
        print(". is found in formula, all data columns will be considered")
        columns = sorted(set(columns) - set([formula.split()[0]]))
        formula = " ".join(formula.split()[:2])
        formula += " + ".join([f"`{col}`" for col in columns])

    return formula


def _check_bounds(lower_limits, upper_limits):
    if (not lower_limits) and (not upper_limits):
        bounds = 0
        bounds_with_zero = 0
        # print("No bounds given")
    elif ((upper_limits is None) and (lower_limits is not None)) or (
        (upper_limits is not None) and (lower_limits is None)
    ):
        raise ValueError("Both upper_limits and lower_limts are needed")
    else:
        if (not isinstance(lower_limits, list)) or (not isinstance(upper_limits, list)):
            raise TypeError("lower_limits and upper_limits must be lists")
        if not (len(lower_limits) == len(upper_limits)):
            raise IndexError("Length of lower_limits and upper_limits must be same")
        non_zero_check_flag = False
        for i in range(len(lower_limits)):
            if lower_limits[i] >= upper_limits[i]:
                raise ValueError(
                    "lower_limit cant be greater than equals to upper_limit"
                )
            if (lower_limits[i] == 0) or (upper_limits[i] == 0):
                pass
            elif not ((0 > lower_limits[i]) and (0 < upper_limits[i])):
                non_zero_check_flag = True
        bounds = 1
        bounds_with_zero = int(not non_zero_check_flag)
    return bounds, bounds_with_zero
