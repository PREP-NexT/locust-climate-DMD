"""
This script is under development.
"""

from analysis_utils import *


#############################  Compute spatial correlation between locust and climate mode magnitude ################################

# Top 10 countries:
cty_code = [435, 670, 750, 625, 770, 501, 679, 436, 615, 600]
cty_name = ['Mauritania', 'Saudi Arabia', 'India', 'Sudan', 'Pakistan', 'Kenya', 'Yemen', 'Niger', 'Algeria', 'Morocco']


#### Compute spatial correlation
cor_matrix = []
[cor_matrix.append([]) for i in range(len(cty_code))]
pvalue_matrix = []
[pvalue_matrix.append([]) for i in range(len(cty_code))]
for cty in cty_code:
    cty_no = cty_code.index(cty)
    # mode_locust, period_locust, mode_climatic, period_climatic are obtained from mrDMD want_1yr section
    # nosea_indices_lo, nosea_indices_c are obtained from load_data_locust and load_data_climatic
    # locust_type, var_name are determined by the user (which locust type and which climatic variable)
    mag_cor, pvalue1, phase_cor, pvalue2 = locust_cond_climatic_phase_sign_country_specific(mode_locust,
                                                                                            period_locust,
                                                                                            mode_climatic,
                                                                                            period_climatic,
                                                                                            nosea_indices_lo,
                                                                                            nosea_indices_c,
                                                                                            locust_type, var_name,
                                                                                            cty_code=[cty],
                                                                                            corr='Spearman',
                                                                                            filter_on=True)
    cor_matrix[cty_no].append(mag_cor)
    pvalue_matrix[cty_no].append(pvalue1)


# copy correlation
# output correlation matrix: rows-country (1-10: mag cor, 11-20: phase corr); column-period from shortest to longest
cor_m = np.asarray(cor_matrix)
pvalue_m = np.asarray(pvalue_matrix)



#############################  Compute inter- and intra-country compound locust risk ################################

#### output to the file
date = '20230705'
txt_path = os.path.join('.', 'RESULTS', 'Phase', f'Phase_{locust_type}_{date}.txt')
txt_file = open(txt_path, "w")


### compute the compound risk
# calculate median phase for each country
phase_10 = []
# Q75-Q25 (intra-country phase range of each country)
phase_10_Q25 = []
phase_10_Q75 = []
# print("____________________________________________________")

for cty in cty_code:
    cty_no = cty_code.index(cty)
    df = locust_cond_climatic_phase_sign_country_specific(mode_locust, period_locust, mode_climatic=None,
                                                            period_climatic=None, nosea_indices_lo=nosea_indices_lo,
                                                            nosea_indices_c=None, locust_type=locust_type,
                                                            var_name=None, cty_code=[cty], corr=None,
                                                            locust_cond='complete')
    phase = df[f'phase({locust_type} locust)[yr]']
    if phase.empty:
        phase = pd.Series([0])
        # print(f'Empty phase in country {cty} is stored as 0!')
        txt_file.write(f'Empty phase in country {cty} is stored as 0!\n')
    phase_10.append(np.median(phase))
    phase_10_Q25.append(np.quantile(phase, 0.25))
    phase_10_Q75.append(np.quantile(phase, 0.75))
phase_10_Q25 = np.asarray(phase_10_Q25)
phase_10_Q75 = np.asarray(phase_10_Q75)
phase_10_range = (phase_10_Q75 - phase_10_Q25).tolist()
## save as dataframe
phase_data = {'phase_median': phase_10, 'phase_range': phase_10_range, 'cty_code': cty_code}
phase_data = pd.DataFrame(phase_data)

phase_median_array = np.asarray(phase_10).reshape((len(cty_code), 1))
phase_median_array_df = pd.DataFrame(phase_median_array, columns=cty_name)
phase_range_array = np.asarray(phase_10_range).reshape((len(cty_code), 1))
phase_range_array_df = pd.DataFrame(phase_range_array, columns=cty_name)

#### process for inter-country compound risk (45 country pairs)
#### phase difference
## initialization
phase_diff = pd.DataFrame()  #index=phase_median_array_df.index
## iteration
for i in range(len(cty_code)):
    cty1 = cty_name[i]
    # if cty1 != 'Kenya':     # exclude Kenya
    for j in range(i+1, len(cty_code)):
        cty2 = cty_name[j]
        # if cty2 != 'Kenya':    # exclude Kenya
        phase_diff[f'{cty1} & {cty2}'] = np.abs(phase_median_array_df[cty1]-phase_median_array_df[cty2])


