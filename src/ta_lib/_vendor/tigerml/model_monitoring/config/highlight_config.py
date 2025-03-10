COLOR_DICT = {
    "bg_green": "background-color: #90EE90",
    "bg_red": "background-color: #d65f5f",
    "bg_yellow": "background-color: #FFFF00",
}

NUM_FORMAT_DICT = {
    "perc_format": "{:,.2%}",
    "roundoff_2dec_format": "{:.2f}",
    "roundoff_3dec_format": "{:.3f}",
    "roundoff_4dec_format": "{:.4f}",
    "count_format": "{:,.0f}",
}

COLUMN_FORMATS = {
    "count_base": "count_format",
    "count_current": "count_format",
    "perc_base": "perc_format",
    "perc_current": "perc_format",
    "psi": "roundoff_4dec_format",
    "dsi": "roundoff_4dec_format",
    "chisquare_stats": "roundoff_2dec_format",
    "chisquare_pvalue": "roundoff_2dec_format",
    "perc_mean_diff": "perc_format",
    "perc_std_diff": "perc_format",
    "perc_count_diff(0)": "perc_format",
    "perc_count_diff(1)": "perc_format",
    "count": "count_format",
    "mode_freq": "count_format",
    "mode_freq_pct": "perc_format",
    "mean": "roundoff_2dec_format",
    "std": "roundoff_2dec_format",
    "min": "roundoff_2dec_format",
    "25%": "roundoff_2dec_format",
    "50%": "roundoff_2dec_format",
    "75%": "roundoff_2dec_format",
    "max": "roundoff_2dec_format",
}
