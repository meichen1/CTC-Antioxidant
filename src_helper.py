


# Create aggregation functions
def sustain_change_to_1(series, set_date=15):
    """Return the first row number (1-based) where the value changes from 0 to 1 and stays 1 or missing"""
    # Reset index to get sequential numbering within the group
    series_reset = series.reset_index(drop=True)
    
    if series_reset.empty:
        return None

    # Find positions where value equals 1
    ones_positions = series_reset[series_reset == 1].index
    
    if len(ones_positions) == 0:
        # Return last position of non-null value
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    # Check each position where value is 1
    for pos in ones_positions:
        # Check if all subsequent values are either 1 or NaN (missing)
        subsequent_values = series_reset[pos+1:]
        if len(subsequent_values) == 0:  # No records after
            return pos + 1
        
        # Check if all subsequent values are 1 or NaN
        valid_subsequent = subsequent_values.isna() | (subsequent_values == 1)
        if valid_subsequent.all():
            return pos + 1
        
        # # Check if all subsequent values are a pattern of all 1s followed by 0s
        # valid_values = subsequent_values.dropna()
        # if not valid_values.empty:
        #     # Convert values to list to check pattern
        #     values_list = valid_values.tolist()
        #     # Find first 0 if it exists
        #     first_zero_idx = values_list.index(0)
        #     # Check if all values before first 0 are 1s and all after are 0s
        #     if all(x == 1 for x in values_list[:first_zero_idx]) and all(x == 0 for x in values_list[first_zero_idx:]):
        #         return pos + 1
        
    # If no sustained recovery found, return len(series) + 1
    return set_date if len(ones_positions) > 0 else None



def first_change_to_1(series, set_date=15):
    """Return the first row number (1-based) where the value changes from 0 to 1 and stays 1 or missing"""
    # Reset index to get sequential numbering within the group
    series_reset = series.reset_index(drop=True)
    
    if series_reset.empty:
        return None

    if len(series_reset[series_reset == 1].index) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    return series_reset[series_reset == 1].index.min() + 1


def ret_indexof1(series):
    """Return the 1-based index of the first occurrence of 1 in the series."""
    # Find the first position of 1
    series_reset = series.reset_index(drop=True)
    pos = series_reset[series_reset == 1].index.tolist()
    pos_zero = series_reset[series_reset == 0].index.tolist()
    return pos, pos_zero


def ret_series(series):
    """Return the series as is"""
    return series.to_list()
    
    
def ret_first_alleviation(series, set_date=15):
    """Return the first row number where the value changes from >1 to <=1"""
    series_reset = series.reset_index(drop=True)
    if series_reset.empty:
        return None
    alleviation_positions = series_reset[series_reset <= 1].index
    if len(alleviation_positions) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    return alleviation_positions.min() + 1


def ret_sustain_alleviation(series, set_date=15):
    """
    Return the first row number where the value changes from >1 to <=1 and 
    subsequent values are mostly <=1 or missing. If no such sustained alleviation is found
    return the first alleviation or None.
    """
    series_reset = series.reset_index(drop=True)
    if series_reset.empty:
        return None

    alleviation_positions = series_reset[series_reset <= 1].index
    if len(alleviation_positions) == 0:
        last_valid = series_reset.last_valid_index()
        return ((last_valid + 1) if last_valid < 13 else set_date) if last_valid is not None else None

    for pos in alleviation_positions:
        subsequent_values = series_reset[pos+1:]
        if len(subsequent_values) == 0:
            return pos + 1
        valid_subsequent = subsequent_values.isna() | (subsequent_values <= 1)
        if valid_subsequent.all():
            return pos + 1
        
    # If no sustained alleviation found, return len(series) + 1
    return set_date if len(alleviation_positions) > 0 else None

