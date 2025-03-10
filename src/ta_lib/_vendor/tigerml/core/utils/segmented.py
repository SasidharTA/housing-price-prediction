def get_segment_filter(data, segment_by, segment):
    filter = None
    assert len(segment_by) == len(segment)  # for testing mostly used
    for idx, segment_col in enumerate(segment_by):
        if filter is None:
            filter = data[segment_col] == segment[idx]
        else:
            filter &= data[segment_col] == segment[idx]
    return filter


def get_segment_from_df(df, seg_cols):
    assert all(
        [df[col].nunique() == 1 for col in seg_cols]
    ), "Passed df has multiple segments"
    return [df[col].unique().tolist()[0] for col in seg_cols]


def calculate_all_segments(data, segment_by):
    return data[segment_by].drop_duplicates().values.tolist()
