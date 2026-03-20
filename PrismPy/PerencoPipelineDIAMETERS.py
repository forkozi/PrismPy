from pathlib import Path
import pandas as pd

pipe_diam_path = Path(r'Perenco\pipelines.csv')

MM_PER_IN = 25.4
IN_PER_M = 1/0.0254

df = pd.read_csv(pipe_diam_path)
od = df['ODsteel_less_coating_in']
coat_corosion_in = df['CorrosionCoating_mm'] / MM_PER_IN
coat_weight_in = df['WeightCoating_mm'] / MM_PER_IN

df['coatings_in'] = (coat_corosion_in + coat_weight_in).round(2)
df['navimodel_diam_in'] = (od + df['coatings_in']).round(2)
df['navimodel_diam_m'] = ((od + df['coatings_in']) / IN_PER_M).round(3)
df.to_html(r'Perenco\pipelines.html')
print(df)