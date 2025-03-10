import numpy as np
import pandas as pd
from tigerml.core.utils import time_now_readable

from .Report import ExcelComponentGroup, ExcelDashboard, ExcelReport


def create_excel_report(
    contents, columns=2, name="", path="", split_sheets=False, **kwargs
):
    if not name:
        name = "report_at_{}".format(time_now_readable())
    report = ExcelReport(name, file_path=path)
    have_plot = False
    n_rows = 100

    if 'excel_params' in kwargs:
        excel_param = kwargs.pop("excel_params")
        if 'have_plot' in excel_param:
            have_plot = excel_param['have_plot']
        if 'n_rows' in excel_param:
            n_rows = excel_param['n_rows']

    contents = process_report_content(contents, index_sheet=True, have_plot=have_plot, n_rows=n_rows)
    if split_sheets:
        for content in contents:
            if isinstance(contents, dict):
                content_name = content
                content = contents[content_name]
            else:
                content_name = "Sheet1"
            report.append_dashboard(
                create_excel_dashboard(content, name=content_name, columns=columns)
            )
    else:
        report.append_dashboard(
            create_excel_dashboard(contents, name="Sheet1", columns=columns)
        )
    report.save()


def create_excel_dashboard(contents, name="", columns=2, flatten=False):
    dash = ExcelDashboard(name=name)
    cg = create_component_group(contents, dash, columns=columns, flatten=flatten)
    dash.append(cg)
    return dash


def group_components(components, dashboard, name="", columns=2, flatten=False):
    cg = ExcelComponentGroup(dashboard, name=name, columns=columns)
    temp_cg = cg
    for component in components:
        if isinstance(component, tuple):
            # import pdb
            # pdb.set_trace()
            import copy

            old_cg = copy.deepcopy(cg)
            old_cg.name = ""
            cg = ExcelComponentGroup(dashboard, name=name, columns=1)
            cg.append(old_cg)
            current_cg = group_components(
                component[1], dashboard, component[0], columns=columns, flatten=flatten
            )
            cg.append(current_cg)
            temp_cg = ExcelComponentGroup(dashboard, name="", columns=2)
        else:
            temp_cg.append(component)
    if cg != temp_cg:
        cg.append(temp_cg)
    return cg


def create_component_group(contents, dashboard, name="", columns=2, flatten=False):
    from ..helpers import create_components

    components = create_components(contents, flatten=flatten, format="xlsx")
    cg = group_components(
        components, dashboard, name=name, columns=columns, flatten=flatten
    )
    return cg


# def create_components(contents, flatten=False):
# 	components = []
# 	for content in contents:
# 		if isinstance(contents, dict):
# 			content_name = content
# 			content = contents[content_name]
# 		else:
# 			content_name = None
# 		if isinstance(content, str):
# 			component = ExcelText(content, name=content_name)
# 		elif str(content.__class__.__module__).startswith('tigerml.core.reports.contents'):
# 			component = get_component_in_format(content, format='xlsx')
# 		elif isinstance(content, ExcelComponentGroup)
# 	    	or isinstance(content, ExcelComponent):
# 			component = content
# 		elif isinstance(content, pd.DataFrame) or isinstance(content, Styler):
# 			component = ExcelTable(content, title=content_name)
# 		elif type(content).__module__.startswith('holoviews')
# 	    	or type(content).__module__.startswith('hvplot') or \
# 			type(content).__module__.startswith('bokeh')
# 	 		or type(content).__module__.startswith('plotly'):
# 			component = ExcelImage(content, name=content_name)
# 		elif isinstance(content, Iterable):
# 			if flatten:
# 				component = create_components(content, flatten=True)
# 			else:
# 				component = (content_name, create_components(content, flatten=False))
# 		else:
# 			component = ExcelImage(content, name=content_name)
# 		# if isinstance(component, list):
# 		# 	components += component
# 		# else:
# 		components.append(component)
# 		if flatten:
# 			from tigerml.core.utils import flatten_list
# 			components = flatten_list(components)
# 	return components

def process_report_content(contents, index_sheet=True, have_plot=True, n_rows=100):

    def flatten(d, parent_key='', sep=' > '):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_dict = flatten(contents)
    keys_to_pop = []
    for key_ in flat_dict.keys():
        if isinstance(flat_dict[key_], str):
            keys_to_pop += [key_]

        elif isinstance(flat_dict[key_], list):
            processed_list = []
            str_in_list = 0
            for item in flat_dict[key_]:
                if isinstance(item, str):
                    str_in_list += 1
                elif 'holoviews' in str(type(item)):
                    if have_plot:
                        processed_list += [{'data': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True).fillna(''), 'plot': item}]
                    else:
                        processed_list += [{'': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True).fillna('')}]
                elif 'matplotlib' in str(type(item)):
                    processed_list += [{'data': '',
                                        'plot': item}]
                elif 'HTMLTable' in str(type(item)):
                    processed_list += [{'': item.data}]
                elif isinstance(item, pd.DataFrame):
                    processed_list += [{'': item}]
                elif isinstance(item, dict):
                    new_item = {}
                    for item_key in item.keys():
                        new_item[key_+'_'+str(item_key)] = item[item_key]
                    processed_list += [process_report_content(new_item, index_sheet=False)]
                else:
                    processed_list += [item]
            if str_in_list == len(flat_dict[key_]):
                keys_to_pop += [key_]
            else:
                flat_dict[key_] = processed_list

        elif isinstance(flat_dict[key_], tuple):
            processed_list = []
            for item in flat_dict[key_]:
                if 'holoviews' in str(type(item)):
                    try:
                        if have_plot:
                            processed_list += [{'data': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True).fillna(''), 'plot': item}]
                        else:
                            processed_list += [{'data': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True).fillna('')}]
                    except ValueError:
                        if have_plot:
                            processed_list += [{'data': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True), 'plot': item}]
                        else:
                            processed_list += [{'data': extract_hv_plot_data(item, n_rows=n_rows).reset_index(drop=True)}]
                elif 'matplotlib' in str(type(item)):
                    processed_list += [{'data': '',
                                        'plot': item}]
                elif 'HTMLTable' in str(type(item)):
                    processed_list += [{'data': item.data,
                                        'plot': ''}]
                elif 'Table.Table' in str(type(item)):
                    processed_list += [{'data': item.data.reset_index(),
                                        'plot': ''}]
                elif isinstance(item, pd.DataFrame):
                    processed_list += [{'data': item, 'plot': ''}]
                elif isinstance(item, dict):
                    new_item = {}
                    for item_key in item.keys():
                        new_item[key_+'_'+str(item_key)] = item[item_key]
                    processed_list += [process_report_content(new_item, index_sheet=False)]
                else:
                    processed_list += [item]

            combined_dict = {'data': [], 'plot': []}
            for dict_ in processed_list:
                if isinstance(dict_['data'], pd.DataFrame):
                    combined_dict['data'] += [dict_['data']]
                if have_plot:
                    if dict_['plot']:
                        combined_dict['plot'] += [dict_['plot']]

            if combined_dict['data']:
                idx = 0
                for df_ in combined_dict['data']:
                    df_.columns = [str(col) + '_' + str(idx) for col in df_.columns]
                    idx += 1

                combined_df = pd.concat(combined_dict['data'], axis=1)
                combined_df = combined_df.T.drop_duplicates().T
                combined_dict['data'] = combined_df.reset_index(drop=True).fillna('')
            if not have_plot:
                combined_dict.pop('plot')
                combined_dict[''] = combined_dict['data']
                combined_dict.pop('data')
            flat_dict[key_] = combined_dict

        elif 'holoviews' in str(type(flat_dict[key_])):
            plot = flat_dict[key_]
            if have_plot:
                flat_dict[key_] = {'data': extract_hv_plot_data(plot, n_rows=n_rows).reset_index(drop=True).fillna(''),
                                   'plot': plot}
            else:
                flat_dict[key_] = {'': extract_hv_plot_data(plot, n_rows=n_rows).reset_index(drop=True).fillna('')}

        elif 'matplotlib' in str(type(flat_dict[key_])):
            plot = flat_dict[key_]
            flat_dict[key_] = {'data': '',
                               'plot': plot}

        elif 'HTMLTable' in str(type(flat_dict[key_])):
            flat_dict[key_] = {'': flat_dict[key_].data}

        elif isinstance(flat_dict[key_], pd.DataFrame):
            flat_dict[key_] = {'': flat_dict[key_]}

    for key_ in keys_to_pop:
        flat_dict.pop(key_)

    if not index_sheet:
        return flat_dict
    idx = 1
    flat_dict_new = {'index_sheet': []}
    index_dic = {'Sheet_No': [], 'content': []}
    for key_ in flat_dict.keys():
        index_dic['Sheet_No'] += ['Sheet' + str(idx)]
        index_dic['content'] += [key_]
        flat_dict_new['Sheet' + str(idx)] = flat_dict[key_]
        idx += 1
    index_df = pd.DataFrame(index_dic)
    flat_dict_new['index_sheet'] = {'': index_df.reset_index(drop=True)}
    # remove_keys = []
    # for key_ in flat_dict_new.keys():
    #     if key_ != 'index_sheet':
    #         remove_keys += [key_]
    # for key_ in remove_keys:
    #     flat_dict_new.pop(key_)
    # for i in range(19,20):flat_dict_new.pop('Sheet'+str(i))
    return flat_dict_new


def extract_hv_plot_data(plot, n_rows=100):

    if isinstance(plot.data, pd.DataFrame):
        if plot.__class__.__name__ == "Distribution":
            data = plot.data[[plot._kdims_param_value[0]._name_param_value]]
            if len(data) > n_rows:
                data = data.sample(n=n_rows)
            return data
        elif plot.__class__.__name__ in ["Scatter", 'HexTiles', 'Points']:
            data = plot.data
            if len(data) > n_rows:
                data = data.sample(n=n_rows)
            return data
        elif plot.__class__.__name__ in ["BoxWhisker", "Violin"]:
            req_cols = []
            if plot._kdims_param_value and plot._kdims_param_value[0]._name_param_value in plot.data.columns:
                req_cols += [plot._kdims_param_value[0]._name_param_value]
            if plot._vdims_param_value and plot._vdims_param_value[0]._name_param_value in plot.data.columns:
                req_cols += [plot._vdims_param_value[0]._name_param_value]
            data = plot.data[req_cols]
            if len(data) > n_rows:
                data = data.sample(n=n_rows)
            return data
        return plot.data
    elif plot.__class__.__name__ in ["HLine", "VLine"]:
        return pd.DataFrame({plot.__class__.__name__: [plot.data]})
    elif isinstance(plot.data, dict):
        if plot.__class__.__name__ == 'HoloMap':
            df_list = []
            for key_ in plot.data.keys():
                df = extract_hv_plot_data(plot.data[key_], n_rows=n_rows)
                df[plot._kdims_param_value[0]._name_param_value] = key_[0]
                df_list += [df]
            data = pd.concat(df_list)

        elif plot.__class__.__name__ == 'DynamicMap':
            try:
                df_list = []
                for key_ in plot._kdims_param_value[0].values:
                    df = extract_hv_plot_data(plot[key_], n_rows=n_rows)
                    df[plot._kdims_param_value[0]._name_param_value] = key_
                    df_list += [df]
                data = pd.concat(df_list)
            except:
                if len(plot.data) == 1:
                    try:
                        for key2 in plot.data.keys():
                            data = plot.data[key2].data
                            pca_dict = {'PC1': data['PC1'], 'PC2': data['PC2']}
                            data = pd.DataFrame(pca_dict)
                    except:
                        print('other than PCA inside DynamicMap', type(plot))
                else:
                    data = pd.DataFrame({' ': ['']})
                    print('new type inside DynamicMap', type(plot))

        elif plot.__class__.__name__ == 'NdOverlay':
            df_list = []
            for key_ in plot.data.keys():
                df = extract_hv_plot_data(plot.data[key_], n_rows=n_rows)
                df[plot._kdims_param_value[0]._name_param_value] = key_[0]
                df_list += [df]
            data = pd.concat(df_list)

        elif plot.__class__.__name__ == 'Layout':
            df_list = []
            for key_ in plot.data.keys():
                df = extract_hv_plot_data(plot.data[key_], n_rows=n_rows)
                # df[plot._name_param_value] = str(key_)

                import holoviews as hv
                bokeh_renderer = hv.renderer("bokeh")
                bokeh_plot = bokeh_renderer.get_plot(plot.data[key_]).state

                try:
                    # df[plot._name_param_value] = bokeh_plot.title.text
                    if bokeh_plot.title.text:
                        df['Layout'] = bokeh_plot.title.text
                    else:
                        df['Layout'] = str(key_)
                except:
                    if bokeh_plot.__class__.__name__ == 'Column':
                        # df[plot._name_param_value] = bokeh_plot.children[0].text.split('>')[1].split('<')[0]
                        df['Layout'] = bokeh_plot.children[0].text.split('>')[1].split('<')[0]
                    else:
                        df['Layout'] = str(key_)

                df_list += [df]
            data = pd.concat(df_list)

        elif plot.__class__.__name__ == 'Overlay':
            df_list = []
            type_list = []
            for key_ in plot.data.keys():
                df = extract_hv_plot_data(plot.data[key_], n_rows=n_rows)
                # df[plot._name_param_value] = str(key_)
                df = df.copy()
                df.columns = [col + '_' + key_[1] for col in df.columns]
                df_list += [df]
                type_list += [plot.data[key_].__class__.__name__]
            # if len(np.unique(type_list)) > 1:
            #     for idx in range(len(df_list)):
            #         df_list[idx].columns = [col+type_list[idx] for col in df_list[idx].columns]
            data = pd.concat(df_list, axis=1)
            data = data.T.drop_duplicates().T

        elif plot.__class__.__name__ == 'Histogram' and isinstance(plot.data, dict):
            data = plot.dframe()

        elif plot.__class__.__name__ == 'HeatMap' and isinstance(plot.data, dict):
            data = pd.DataFrame(plot.data["value"],
                                index=plot.data["index"],
                                columns=plot.data["columns"]).reset_index()

        else:
            print('new plot type', type(plot))
        return data
