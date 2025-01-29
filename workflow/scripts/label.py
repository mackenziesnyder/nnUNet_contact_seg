import pandas as pd
import numpy as np

# def calculate_distance_to_vector(point, vector_point, vector_direction):
#     vector_point = np.array(vector_point)
#     vector_direction = np.array(vector_direction)
#     point = np.array(point)

#     unit_vector = vector_direction / np.linalg.norm(vector_direction)
#     vector_to_point = point - vector_point
#     distance_along_vector = np.dot(vector_to_point, unit_vector)

#     return distance_along_vector

def create_electrode_list(df_te):
    electrodes = []
    for i in range(0, len(df_te), 2):
        target_point = df_te.iloc[i][[1, 2, 3]].values
        entry_point = df_te.iloc[i + 1][[1, 2, 3]].values
        electrode_label = df_te.iloc[i][11]

        electrodes.append({
            'entry_point': entry_point,
            'target_point': target_point,
            'contacts': [],
            'elec_label': electrode_label
        })
    return electrodes


def df_to_fcsv(input_df, output_fcsv):
	with open(output_fcsv, 'w') as fid:
		fid.write("# Markups fiducial file version = 4.11\n")
		fid.write("# CoordinateSystem = 0\n")
		fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")

	out_df={'node_id':[],'x':[],'y':[],'z':[],'ow':[],'ox':[],'oy':[],'oz':[],
		'vis':[],'sel':[],'lock':[],'label':[],'description':[],'associatedNodeID':[]
	}

	for idx,ifid in input_df.iterrows():
		out_df['node_id'].append(idx+1)
		out_df['x'].append(ifid.iloc[0])
		out_df['y'].append(ifid.iloc[1])
		out_df['z'].append(ifid.iloc[2])
		out_df['ow'].append(0)
		out_df['ox'].append(0)
		out_df['oy'].append(0)
		out_df['oz'].append(0)
		out_df['vis'].append(1)
		out_df['sel'].append(1)
		out_df['lock'].append(1)
		out_df['label'].append(str(ifid.iloc[3]))
		out_df['description'].append('NA')
		out_df['associatedNodeID'].append('')

	out_df=pd.DataFrame(out_df)
	out_df.to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False, float_format = '%.3f')

if __name__ == "__main__":
    df_te = pd.read_csv(snakemake.input['planned_fcsv'], skiprows = 3, header = None)
    df_contacts = pd.read_csv(snakemake.input['coords'], skiprows = 3, header = None)

    contact_array = df_contacts[[1, 2, 3]].values
    te_coords = df_te[[1, 2, 3, 11]].values

    electrode_list = create_electrode_list(df_te)
    grouped = assign_contacts_to_electrodes(electrode_list, contact_array, 3.0)
    labelled_contacts = label_multiple_electrodes(electrode_list)
    labelled_df = convert_to_df(labelled_contacts)

    df_to_fcsv(labelled_df, snakemake.output['labelled_coords'])        

