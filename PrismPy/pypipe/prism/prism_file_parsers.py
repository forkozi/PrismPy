import re
import json
from pathlib import Path
import numpy as np
import pandas as pd


class AnomaliesParser:

    def __init__(self, anoms_path):
        self.path = anoms_path
        with open(self.path) as f:
            self.file_txt = f.read()
        self.regex = ''.join([
            r'(\w+),(\w+),(\w+),([A-Z0-9- ]+)\n',   # required fields
            r'((?:^#[^\n]+\n)+)'                    # optional fields
        ])
        self.optional_fields = self.get_optional_field_names()

    @staticmethod
    def get_optional_field_names():
        opt_fields_path = Path('PRISM\prism_anomaly_options.json')
        with open(opt_fields_path) as f:

            return json.load(f)

    def parse_optionals(self, optionals):
        optionals = optionals.strip().split('\n')
        
        return {self.optional_fields[o[0:3]]:o[3:] for o in optionals}

    def parse(self):
        anomalies = re.findall(self.regex, self.file_txt, re.MULTILINE)
        df_data = []
        for anomaly in anomalies:
            data = {
                'Asset': anomaly[0],
                'Pipeline': anomaly[1],
                'AnomalyNum': anomaly[2],
                'Images': re.split(r'\s+', anomaly[3]),
            }
            opts = self.parse_optionals(anomaly[4])
            df_data.append({**data, **opts})

        return pd.DataFrame(df_data)


class EventsParser:

    def __init__(self, events_path):
        self.path = events_path
        with open(self.path) as f:
            self.file_txt = f.read()
        self.regex = ''.join([
            r'(\w+),(\w+),(\w+),([A-Z0-9- ]+)\n',
            r'((?:^#[^\n]+\n)+)'
        ])
        self.field_specs = self.get_fields_spec()
        self.dtyps = {
            'Numeric': np.float64,
            'Integer': np.int64,
            'Character': str,
            'Boolean': bool,
        }


    def get_fields_spec(self):
        fields_path = Path('PRISM\prism_events_fields.csv')

        return pd.read_csv(fields_path)

    def parse(self):

        return pd.read_csv(self.path, header=None, 
                           names=self.field_specs['FieldDescription'])


def list_file_data(files):
    ftypes = {
        'anomalies': AnomaliesParser,
        'events': EventsParser,
    }
    output_fpaths = []
    for ftype, fpaths in files.items():
        dfs = []
        for fpath in fpaths:
            df = ftypes[ftype](fpath).parse()
            df['SourceFile'] = fpath.relative_to(pipeline_path)
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        out_path = Path(f'PERENCO_{ftype.upper()}.csv')
        df.to_csv(out_path)
        output_fpaths.append(out_path)

    return output_fpaths


if __name__ == '__main__':
    onedrive_path = Path(r'C:\Users\n.forfinski-sarkozi\OneDrive - Fugro')
    pipeline_path = onedrive_path / 'Projects\Perenco_CP\Deliverables SB'

    output_fpaths = list_file_data({
        'anomalies': pipeline_path.rglob(r'anoms*'),
        'events': pipeline_path.rglob(r'PL*.dat')
        })
    print(output_fpaths)

