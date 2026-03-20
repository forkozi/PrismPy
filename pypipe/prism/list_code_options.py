import json
import pandas as pd

df = pd.read_csv(r'PRISM\prism_event_codes.csv')
df['EventClassification'] = df[[
    'PrimaryDescription', 
    'SecondaryDescription', 
    'TertiaryDescription'
    ]].apply(tuple, axis=1).astype(str)
df = df.set_index(['Primary', 'Secondary', 'Tertiary'])
df['PST_idx'] = df.index.str


data = df.stack(level=[0]).to_dict()
d = {}
for t, v in data.items():
    e = d.setdefault(t[0], {})
    for k in t[1:-1]:
        e = e.setdefault(k, {})
    e[t[-1]] = v
print(d)

with open('prism_event_codes.json', 'w') as j:
    json.dump(d, j, indent=2)


# for code, indices in df.groupby('PrimaryDescription').groups.items():
#     print(code)
#     df_ = df.iloc[indices][['Primary', 'PrimaryDescription']]
#     print(df)
#     df_ = df_.drop_duplicates(['Primary', 'PrimaryDescription'])
#     print(df_)

