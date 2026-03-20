from datetime import datetime
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


class NaviModelPipelineProject:

    def __init__(self, pipeline, meta):
        self.pipeline = pipeline
        self.asset = meta.Asset
        self.proj_dir = Path(meta.NaviModelProjDir)
        try:
            self.cpp_path = Path(meta.CPP_Path)
        except TypeError as e:
            self.cpp_path = None
        self.events_dir = self.proj_dir / 'Export'
        self.eff_dir = self.proj_dir / 'Data/Pipes'
        self.images_dir = self.proj_dir / 'images'
        self.prism_dir = Path(meta.PRISM_Dir)
        self.datetime = self.get_datetime_obj(meta.MidDateTime)
        self.yy = self.datetime.strftime('%y')
        self.yyyy = self.datetime.strftime('%Y')

        # events info
        events_filename = f'{self.pipeline}_{self.yy}.dat'
        self.events_path = self.prism_dir / self.asset / events_filename
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
    
        # anomalies info
        anomalies_file_name = f'anoms_{self.yy}.dat'
        self.anomalies_path = self.prism_dir / self.asset / anomalies_file_name
        self.anomalies_path.parent.mkdir(parents=True, exist_ok=True)
        self.prism_imgs_dir = self.prism_dir / 'images' / self.asset / self.yyyy
        self.prism_imgs_dir.parent.mkdir(parents=True, exist_ok=True)
    
    def get_datetime_obj(self, mid_datetime):
        date_format = r'%Y/%m/%d %H:%M:%S'
        
        return datetime.strptime(mid_datetime, date_format)

    def export_events(self):
        self.events = NaviModelEvents(self)
        self.start_stop_survey_events = NaviModelSurveyStartStopEvents(self)
        self.burial_events = NaviModelEffBurialEvents(self)
        self.nav_markers = NaviModelEffNavMarkers(self)
        self.events.add_events(self.start_stop_survey_events)
        self.events.add_events(self.nav_markers)
        self.events.add_events(self.nav_markers)
        self.events.add_events(self.burial_events)
        self.events.to_prism(self.events_path)

        return self.events_path

    def export_anomalies(self):
        opts_to_include = [1, 2, 13, 14, 15]
        self.events.export_anomalies(self.anomalies_path, opts_to_include)
        
        return self.anomalies_path
    
    def package_images(self):
        pass


class PrismEvents:

    def __init__(self, navi_proj):
        self.df = pd.DataFrame([])
        self.cpp = pd.DataFrame([])
        if navi_proj.cpp_path:
            try:
                self.cpp = pd.read_excel(navi_proj.cpp_path, 
                                         sheet_name=navi_proj.pipeline)
            except ValueError as e:
                pass
        self.field_formats = {
            'KP': '{0:0.4f}',  # Numeric,4
            'Easting': '{0:0.3f}',  # Numeric,3
            'Northing': '{0:0.3f}',  # Numeric,3
            'Time': '{0}',  # Integer,0
            'Date': '{0}',  # Integer,0
            'DCC': '{0:0.3f}',  # Numeric,
            'SD': '{0}',  # Numeric,1
            'Comments': '{0}',  # Character,''
            'Video': '{0}',  # Character,1
            'Anomaly': '{0}',  # Integer,0
            'Bathy': '{0}',  # Numeric,2
            'DOB': '{0}',  # Numeric,2
            'ContactCP': '{0:4.0f}',  # Integer,
            'ContinuousCP': '{0:4.0f}',  # Integer,0
            'CoverLocal': '{0}',  # Numeric,2
            'CoverMSB': '{0}',  # Numeric,2
            'Code1': '{0}',  # Integer,0
            'Code2': '{0}',  # Integer,0
            'Code3': '{0}',  # Integer,0
            'Span': '{0}',  # Boolean,0
            'Exposure': '{0}',  # Boolean,0
            'Movement': '{0}',  # Boolean,0
            'WghtCt': '{0}',  # Boolean,0
            'Stab': '{0}',  # Boolean,0
            'SpanLength': '{0}',  # Numeric,2
            'SpanLengthOdom': '{0}',  # Numeric,2
            'SpanHeight': '{0}',  # Numeric,2
            'SpanDEI': '{0}',  # Character,'',''
            'Location': '{0}',  # Integer,0
            'Offset': '{0}',  # Numeric,2
            'DimX': '{0}',  # Numeric,2
            'DimY': '{0}',  # Numeric,2
            'DimZ': '{0}',  # Numeric,2
            'Anomlink': '{0}',  # Character,'',''
            'VideoCount': '{0}',  # Integer,0
            'UniqueID': '{0}',  # Integer,0
            'Dummy3': '{0}',  # Numeric,''
            'Dummy4': '{0}',  # Numeric,''
            'DummyString': '{0}',  # Character,''
        }

    def to_prism(self, prism_path):
        format_func = lambda x: x.apply(self.field_formats[x.name].format)
        df = self.df.apply(format_func)
        df.to_csv(prism_path, index=False)
        return prism_path


class NaviModelSurveyStartStopEvents(PrismEvents):

    def __init__(self, navi_proj):
        super().__init__(navi_proj)
        self.navi_proj = navi_proj
        self.eff_paths = navi_proj.eff_dir.rglob(r'*.eff')
        self.flags = {}
        src_df = self.parse_eff_files().sort_values('#KP(km)')
        self.src_df = src_df.bfill().ffill()[::src_df.shape[0]-1]
        self.data = self.populate_data()
        self.df = pd.DataFrame(self.data)

    def populate_data(self):
        if not self.src_df.empty:
            return {
                'KP': self.get_KP(),
                'Easting': self.get_Easting(),
                'Northing': self.get_Northing(),
                'Time': self.get_Time(),
                'Date': self.get_Date(),
                'DCC': self.get_DCC(),
                'SD': self.get_SD(),
                'Comments': self.get_Comments(),
                'Video': self.get_Video(),
                'Anomaly': self.get_Anomaly(),
                'Bathy': self.get_Bathy(),
                'DOB': self.get_DOB(),
                'ContactCP': self.get_ContactCP(),
                'ContinuousCP': self.get_ContinuousCP(),
                'CoverLocal': self.get_CoverLocal(),
                'CoverMSB': self.get_CoverMSB(),
                'Code1': self.get_Code1(),
                'Code2': self.get_Code2(),
                'Code3': self.get_Code3(),
                'Span': self.get_Span(),
                'Exposure': self.get_Exposure(),
                'Movement': self.get_Movement(),
                'WghtCt': self.get_WghtCt(),
                'Stab': self.get_Stab(),
                'SpanLength': self.get_SpanLength(),
                'SpanLengthOdom': self.get_SpanLengthOdom(),
                'SpanHeight': self.get_SpanHeight(),
                'SpanDEI': self.get_SpanDEI(),
                'Location': self.get_Location(),
                'Offset': self.get_Offset(),
                'DimX': self.get_DimX(),
                'DimY': self.get_DimY(),
                'DimZ': self.get_DimZ(),
                'Anomlink': self.get_Anomlink(),
                'VideoCount': self.get_VideoCount(),
                'UniqueID': self.get_UniqueID(),
                'Dummy3': self.get_Dummy3(),
                'Dummy4': self.get_Dummy4(),
                'DummyString': self.get_DummyString(),
            }
        else:
            return {}

    def parse_eff_files(self):
        for eff_path in self.eff_paths:
            self.flags[eff_path.stem] = pd.read_csv(eff_path,
                                                    header=5, sep=r'\s+')
        return pd.concat(self.flags.values())

    def get_KP(self):
        return self.src_df['#KP(km)']

    def get_Easting(self):
        return self.src_df['Cover_X']

    def get_Northing(self):
        return self.src_df['Cover_Y']

    def get_Time(self):
        """HHMMSS"""
        return self.navi_proj.datetime.strftime('%H%M%S')

    def get_Date(self):
        """DDMMYYYY"""
        return self.navi_proj.datetime.strftime('%d%m%Y')

    def get_DCC(self):
        return self.src_df['DCC']

    def get_SD(self):
        return ''

    def get_Comments(self):
        return ['Start of MBES Survey', 'End of MBES Survey']

    def get_Video(self):
        return ''

    def get_Anomaly(self):
        return ''

    def get_Bathy(self):
        return ''

    def get_DOB(self):
        return ''

    def get_ContactCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential On'])

    def get_ContinuousCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential Off'])

    def get_CoverLocal(self):
        return ''

    def get_CoverMSB(self):
        return ''

    def get_Code1(self):
        return 12

    def get_Code2(self):
        return [1, 2]  # [start, end]

    def get_Code3(self):
        return 0

    def get_Span(self):
        return ''

    def get_Exposure(self):
        return ''

    def get_Movement(self):
        return ''

    def get_WghtCt(self):
        return ''

    def get_Stab(self):
        return ''

    def get_SpanLength(self):
        return ''

    def get_SpanLengthOdom(self):
        return ''

    def get_SpanHeight(self):
        return ''

    def get_SpanDEI(self):
        return ''

    def get_Location(self):
        return ''

    def get_Offset(self):
        return ''

    def get_DimX(self):
        return ''

    def get_DimY(self):
        return ''

    def get_DimZ(self):
        return ''

    def get_Anomlink(self):
        return ''

    def get_VideoCount(self):
        return ''

    def get_UniqueID(self):
        return ''

    def get_Dummy3(self):
        return ''

    def get_Dummy4(self):
        return ''

    def get_DummyString(self):
        return ''           


class NaviModelEvents(PrismEvents):

    def __init__(self, navi_proj):
        super().__init__(navi_proj)
        self.navi_proj = navi_proj
        self.event_paths = navi_proj.events_dir.rglob(r'*_events_*.csv')
        self.events = {}
        self.src_df = self.parse_event_files()
        self.data = self.populate_data()
        self.df = pd.DataFrame(self.data)

    def populate_data(self):
        if not self.src_df.empty:
            return {
                'KP': self.get_KP(),
                'Easting': self.get_Easting(),
                'Northing': self.get_Northing(),
                'Time': self.get_Time(),
                'Date': self.get_Date(),
                'DCC': self.get_DCC(),
                'SD': self.get_SD(),
                'Comments': self.get_Comments(),
                'Video': self.get_Video(),
                'Anomaly': self.get_Anomaly(),
                'Bathy': self.get_Bathy(),
                'DOB': self.get_DOB(),
                'ContactCP': self.get_ContactCP(),
                'ContinuousCP': self.get_ContinuousCP(),
                'CoverLocal': self.get_CoverLocal(),
                'CoverMSB': self.get_CoverMSB(),
                'Code1': self.get_Code1(),
                'Code2': self.get_Code2(),
                'Code3': self.get_Code3(),
                'Span': self.get_Span(),
                'Exposure': self.get_Exposure(),
                'Movement': self.get_Movement(),
                'WghtCt': self.get_WghtCt(),
                'Stab': self.get_Stab(),
                'SpanLength': self.get_SpanLength(),
                'SpanLengthOdom': self.get_SpanLengthOdom(),
                'SpanHeight': self.get_SpanHeight(),
                'SpanDEI': self.get_SpanDEI(),
                'Location': self.get_Location(),
                'Offset': self.get_Offset(),
                'DimX': self.get_DimX(),
                'DimY': self.get_DimY(),
                'DimZ': self.get_DimZ(),
                'Anomlink': self.get_Anomlink(),
                'VideoCount': self.get_VideoCount(),
                'UniqueID': self.get_UniqueID(),
                'Dummy3': self.get_Dummy3(),
                'Dummy4': self.get_Dummy4(),
                'DummyString': self.get_DummyString(),
            }
        else:
            return {}

    def export_anomalies(self, anomalies_path, opts):
        func = lambda e: PrismAnomaly(e, self.navi_proj).to_prism(opts)
        anomalies = self.src_df.apply(func, axis=1)       
        with open(anomalies_path, 'a') as f:
            try:
                f.writelines('\n\n'.join(anomalies.to_list()))
                f.writelines(['\n', '\n'])
            except AttributeError as e:
                pass

    def add_events(self, nav_markers):
        self.df = pd.concat([self.df, nav_markers.df])

    def parse_event_files(self):
        
        for event_path in self.event_paths:
            etype = event_path.stem.split('_')[-1]
            df =pd.read_csv(event_path)
            df['pipeline_id'] = event_path.stem.split('_')[0]
            self.events[etype] = df

        try:
            return pd.concat(self.events.values())
        except ValueError as e:
            return pd.DataFrame({})

    def get_KP(self):
        return self.src_df['KP (Dynamic)']

    def get_Easting(self):
        return self.src_df['Easting']

    def get_Northing(self):
        return self.src_df['Northing']

    def get_Time(self):
        """HHMMSS"""
        return self.navi_proj.datetime.strftime('%H%M%S')

    def get_Date(self):
        """DDMMYYYY"""
        return self.navi_proj.datetime.strftime('%d%m%Y')

    def get_DCC(self):
        return self.src_df['DCC (Dynamic)']

    def get_SD(self):
        return ''

    def get_Comments(self):
        return self.src_df['Name']

    def get_Video(self):
        return ''

    def get_Anomaly(self):
        return self.src_df['Source (Static)'].astype(str)

    def get_Bathy(self):
        return ''

    def get_DOB(self):
        return ''

    def get_ContactCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential On'])

    def get_ContinuousCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential Off'])

    def get_CoverLocal(self):
        return ''

    def get_CoverMSB(self):
        return ''

    def get_Code1(self):
        func = lambda e: e.split(' ')[0]
        return self.src_df['Classification (Static)'].apply(func)

    def get_Code2(self):
        func = lambda e: e.split(' ')[1]
        return self.src_df['Classification (Static)'].apply(func)

    def get_Code3(self):
        func = lambda e: e.split(' ')[2]
        return self.src_df['Classification (Static)'].apply(func)

    def get_Span(self):
        return ''

    def get_Exposure(self):
        return ''

    def get_Movement(self):
        return ''

    def get_WghtCt(self):
        return ''

    def get_Stab(self):
        return ''

    def get_SpanLength(self):
        return ''

    def get_SpanLengthOdom(self):
        return ''

    def get_SpanHeight(self):
        return ''

    def get_SpanDEI(self):
        return ''

    def get_Location(self):
        return ''

    def get_Offset(self):
        return ''

    def get_DimX(self):
        return ''

    def get_DimY(self):
        return ''

    def get_DimZ(self):
        return ''

    def get_Anomlink(self):
        return ''

    def get_VideoCount(self):
        return ''

    def get_UniqueID(self):
        return ''

    def get_Dummy3(self):
        return ''

    def get_Dummy4(self):
        return ''

    def get_DummyString(self):
        return ''
        

class NaviModelEffNavMarkers(PrismEvents):

    def __init__(self, navi_proj):
        super().__init__(navi_proj)
        self.navi_proj = navi_proj
        self.eff_paths = navi_proj.eff_dir.rglob(r'*.eff')
        self.flags = {}
        self.src_df = self.parse_eff_files()
        self.data = self.populate_data()
        self.df = pd.DataFrame(self.data)[::12]

    def populate_data(self):
        if not self.src_df.empty:
            return {
                'KP': self.get_KP(),
                'Easting': self.get_Easting(),
                'Northing': self.get_Northing(),
                'Time': self.get_Time(),
                'Date': self.get_Date(),
                'DCC': self.get_DCC(),
                'SD': self.get_SD(),
                'Comments': self.get_Comments(),
                'Video': self.get_Video(),
                'Anomaly': self.get_Anomaly(),
                'Bathy': self.get_Bathy(),
                'DOB': self.get_DOB(),
                'ContactCP': self.get_ContactCP(),
                'ContinuousCP': self.get_ContinuousCP(),
                'CoverLocal': self.get_CoverLocal(),
                'CoverMSB': self.get_CoverMSB(),
                'Code1': self.get_Code1(),
                'Code2': self.get_Code2(),
                'Code3': self.get_Code3(),
                'Span': self.get_Span(),
                'Exposure': self.get_Exposure(),
                'Movement': self.get_Movement(),
                'WghtCt': self.get_WghtCt(),
                'Stab': self.get_Stab(),
                'SpanLength': self.get_SpanLength(),
                'SpanLengthOdom': self.get_SpanLengthOdom(),
                'SpanHeight': self.get_SpanHeight(),
                'SpanDEI': self.get_SpanDEI(),
                'Location': self.get_Location(),
                'Offset': self.get_Offset(),
                'DimX': self.get_DimX(),
                'DimY': self.get_DimY(),
                'DimZ': self.get_DimZ(),
                'Anomlink': self.get_Anomlink(),
                'VideoCount': self.get_VideoCount(),
                'UniqueID': self.get_UniqueID(),
                'Dummy3': self.get_Dummy3(),
                'Dummy4': self.get_Dummy4(),
                'DummyString': self.get_DummyString(),
            }
        else:
            return {}

    def parse_eff_files(self):
        for eff_path in self.eff_paths:
            self.flags[eff_path.stem] = pd.read_csv(eff_path,
                                                    header=5, sep=r'\s+')
        return pd.concat(self.flags.values())

    def get_KP(self):
        return self.src_df['#KP(km)']

    def get_Easting(self):
        return self.src_df['Cover_X']

    def get_Northing(self):
        return self.src_df['Cover_Y']

    def get_Time(self):
        """HHMMSS"""
        return self.navi_proj.datetime.strftime('%H%M%S')

    def get_Date(self):
        """DDMMYYYY"""
        return self.navi_proj.datetime.strftime('%d%m%Y')

    def get_DCC(self):
        return self.src_df['DCC']

    def get_SD(self):
        return ''

    def get_Comments(self):
        return 'Nav Marker'

    def get_Video(self):
        return ''

    def get_Anomaly(self):
        return ''

    def get_Bathy(self):
        return ''

    def get_DOB(self):
        return ''

    def get_ContactCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential On'])

    def get_ContinuousCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential Off'])

    def get_CoverLocal(self):
        lin_z = self.src_df['LIn_Z']
        rin_z = self.src_df['RIn_Z']
        cover_z = self.src_df['Cover_Z']

        return np.round(np.average((lin_z, rin_z), axis=0) - cover_z, 2)

    def get_CoverMSB(self):
        lout_z = self.src_df['LOut_Z']
        rout_z = self.src_df['ROut_Z']
        cover_z = self.src_df['Cover_Z']

        return np.round(np.average((lout_z, rout_z), axis=0) - cover_z, 2)

    def get_Code1(self):
        return 0

    def get_Code2(self):
        return 0

    def get_Code3(self):
        return 0

    def get_Span(self):
        """['Exposed', 'Covered', 'Freespan']"""
        exposure_func = lambda f: 1 if f == 'Freespan' else 0
        return self.src_df['Burial'].apply(exposure_func)

    def get_Exposure(self):
        """['Exposed', 'Covered', 'Freespan']"""
        exposure_func = lambda f: 1 if f == 'Exposed' else 0
        return self.src_df['Burial'].apply(exposure_func)

    def get_Movement(self):
        return ''

    def get_WghtCt(self):
        return ''

    def get_Stab(self):
        return ''

    def get_SpanLength(self):
        return ''

    def get_SpanLengthOdom(self):
        return ''

    def get_SpanHeight(self):
        return ''

    def get_SpanDEI(self):
        return ''

    def get_Location(self):
        return ''

    def get_Offset(self):
        return ''

    def get_DimX(self):
        return ''

    def get_DimY(self):
        return ''

    def get_DimZ(self):
        return ''

    def get_Anomlink(self):
        return ''

    def get_VideoCount(self):
        return ''

    def get_UniqueID(self):
        return ''

    def get_Dummy3(self):
        return ''

    def get_Dummy4(self):
        return ''

    def get_DummyString(self):
        return ''           


class NaviModelEffBurialEvents(PrismEvents):

    def __init__(self, navi_proj):
        super().__init__(navi_proj)
        self.prism_codes = {
            'Exposed_start': [7, 1, 0],
            'Exposed_end': [7, 2, 0],
            'Freespan_start': [8, 1, 0],
            'Freespan_end': [8, 2, 0],
        }
        self.navi_proj = navi_proj
        self.eff_paths = navi_proj.eff_dir.rglob(r'*.eff')
        self.flags = {}
        self.src_df = self.parse_eff_files()
        self.data = self.populate_data()
        self.df = pd.DataFrame(self.data)

    def populate_data(self):
        if not self.src_df.empty:
            return {
                'KP': self.get_KP(),
                'Easting': self.get_Easting(),
                'Northing': self.get_Northing(),
                'Time': self.get_Time(),
                'Date': self.get_Date(),
                'DCC': self.get_DCC(),
                'SD': self.get_SD(),
                'Comments': self.get_Comments(),
                'Video': self.get_Video(),
                'Anomaly': self.get_Anomaly(),
                'Bathy': self.get_Bathy(),
                'DOB': self.get_DOB(),
                'ContactCP': self.get_ContactCP(),
                'ContinuousCP': self.get_ContinuousCP(),
                'CoverLocal': self.get_CoverLocal(),
                'CoverMSB': self.get_CoverMSB(),
                'Code1': self.get_Code1(),
                'Code2': self.get_Code2(),
                'Code3': self.get_Code3(),
                'Span': self.get_Span(),
                'Exposure': self.get_Exposure(),
                'Movement': self.get_Movement(),
                'WghtCt': self.get_WghtCt(),
                'Stab': self.get_Stab(),
                'SpanLength': self.get_SpanLength(),
                'SpanLengthOdom': self.get_SpanLengthOdom(),
                'SpanHeight': self.get_SpanHeight(),
                'SpanDEI': self.get_SpanDEI(),
                'Location': self.get_Location(),
                'Offset': self.get_Offset(),
                'DimX': self.get_DimX(),
                'DimY': self.get_DimY(),
                'DimZ': self.get_DimZ(),
                'Anomlink': self.get_Anomlink(),
                'VideoCount': self.get_VideoCount(),
                'UniqueID': self.get_UniqueID(),
                'Dummy3': self.get_Dummy3(),
                'Dummy4': self.get_Dummy4(),
                'DummyString': self.get_DummyString(),
            }
        else:
            return {}

    @staticmethod
    def parse_burial_events(eff_path, etype='Exposed'):
        df = pd.read_csv(eff_path, header=5, sep=r'\s+')
        if etype == 'Exposed':
            df = df[df['Burial']!='Freespan']
        runline_idx = df['BasedOn']=='Runline'
        df.loc[runline_idx,'Burial'] = 'Covered'

        is_shift_eq = df.Burial.shift(-1) == df.Burial
        burial_events = df.loc[is_shift_eq[is_shift_eq==False].index].Burial
        
        i_df = pd.DataFrame(burial_events.index.insert(0,-1))
        start_indices = i_df + 1
        end_indices = list(burial_events.index + 1)
        end_indices[-1] = end_indices[-1] - 1

        num_events = i_df[0][1:].values - i_df[0][:-1].values

        start_idx = start_indices.iloc[:-1][0].to_list()
        end_idx = end_indices  # simply to maintain naming consistency with start

        burial_df = pd.DataFrame(burial_events)
        burial_df['num_events'] = num_events

        def build_events(idx, idx_type):
            df_idx = burial_df.copy()
            df_idx['idx'] = idx
            df_idx['kp'] = df.loc[idx]['#KP(km)'].values
            df_idx['dcc'] = df.loc[idx]['DCC'].values
            df_idx['LIn_Z'] = df.loc[idx]['LIn_Z'].values
            df_idx['RIn_Z'] = df.loc[idx]['RIn_Z'].values
            df_idx['LOut_Z'] = df.loc[idx]['LOut_Z'].values
            df_idx['ROut_Z'] = df.loc[idx]['ROut_Z'].values
            df_idx['Cover_X'] = df.loc[idx]['Cover_X'].values
            df_idx['Cover_Y'] = df.loc[idx]['Cover_Y'].values
            df_idx['Cover_Z'] = df.loc[idx]['Cover_Z'].values
            df_idx['Comment'] = df_idx['Burial'].astype(str) + '_' + idx_type

            return df_idx

        eff_events = pd.concat([
            build_events(start_idx, 'start'),
            build_events(end_idx, 'end')
            ], axis=0).sort_index()
        eff_events.index.name = 'Idx'
        eff_events = eff_events.sort_values(by=['Idx', 'Comment'], 
                                            ascending=[True, False])
        
        return eff_events[eff_events['Burial']==etype]

    @staticmethod
    def gen_burial_events_summary(eff_path):
        df = pd.read_csv(eff_path, header=5, sep=r'\s+')
        # df = df[df['Burial']!='Freespan']
        runline_idx = df['BasedOn']=='Runline'
        df.loc[runline_idx,'Burial'] = 'Covered'

        is_shift_eq = df.Burial.shift(-1) == df.Burial
        burial_events = df.loc[is_shift_eq[is_shift_eq==False].index].Burial
        
        i_df = pd.DataFrame(burial_events.index.insert(0,-1))
        start_indices = i_df + 1
        end_indices = list(burial_events.index + 1)
        end_indices[-1] = end_indices[-1] - 1

        num_events = i_df[0][1:].values - i_df[0][:-1].values

        start_idx = start_indices.iloc[:-1][0].to_list()
        end_idx = end_indices  # simply to maintain naming consistency with start

        burial_df = pd.DataFrame(burial_events)
        burial_df['num_events'] = num_events
        burial_df['start_idx'] = start_idx
        burial_df['end_idx'] = end_indices
        burial_df['start_kp'] = df.loc[start_idx]['#KP(km)'].values
        burial_df['end_kp'] = df.loc[end_idx]['#KP(km)'].values     

        return burial_df

    def parse_eff_files(self):
        for eff_path in self.eff_paths:
            exposures = self.parse_burial_events(eff_path, etype='Exposed')
            freespans = self.parse_burial_events(eff_path, etype='Freespan')
            self.flags[eff_path.stem] = pd.concat([exposures, freespans])

        return pd.concat(self.flags.values()).sort_values(by='kp')

    def get_KP(self):
        return self.src_df['kp']

    def get_Easting(self):
        return self.src_df['Cover_X']

    def get_Northing(self):
        return self.src_df['Cover_Y']

    def get_Time(self):
        """HHMMSS"""
        return self.navi_proj.datetime.strftime('%H%M%S')

    def get_Date(self):
        """DDMMYYYY"""
        return self.navi_proj.datetime.strftime('%d%m%Y')

    def get_DCC(self):
        return self.src_df['dcc']

    def get_SD(self):
        return ''

    def get_Comments(self):
        return self.src_df['Comment']

    def get_Video(self):
        return ''

    def get_Anomaly(self):
        return ''

    def get_Bathy(self):
        return ''

    def get_DOB(self):
        return ''

    def get_ContactCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential On'])

    def get_ContinuousCP(self):
        if self.cpp.empty:
            return np.nan
        else:
            return np.interp(x=self.get_KP(),
                             xp=self.cpp.filter(regex=r'K\.*P\.*').squeeze(),
                             fp=self.cpp['Potential Off'])

    def get_CoverLocal(self):
        lin_z = self.src_df['LIn_Z']
        rin_z = self.src_df['RIn_Z']
        cover_z = self.src_df['Cover_Z']

        return np.round(np.average((lin_z, rin_z), axis=0) - cover_z, 2)

    def get_CoverMSB(self):
        lout_z = self.src_df['LOut_Z']
        rout_z = self.src_df['ROut_Z']
        cover_z = self.src_df['Cover_Z']

        return np.round(np.average((lout_z, rout_z), axis=0) - cover_z, 2)

    def get_Code1(self):
        return self.src_df.Comment.apply(lambda r: self.prism_codes[r][0])

    def get_Code2(self):
        return self.src_df.Comment.apply(lambda r: self.prism_codes[r][1])

    def get_Code3(self):
        return self.src_df.Comment.apply(lambda r: self.prism_codes[r][2])

    def get_Span(self):
        """['Exposed', 'Covered', 'Freespan']"""
        exposure_func = lambda f: 1 if f == 'Freespan' else 0
        return self.src_df['Burial'].apply(exposure_func)

    def get_Exposure(self):
        """['Exposed', 'Covered', 'Freespan']"""
        exposure_func = lambda f: 1 if f == 'Exposed' else 0
        return self.src_df['Burial'].apply(exposure_func)

    def get_Movement(self):
        return ''

    def get_WghtCt(self):
        return ''

    def get_Stab(self):
        return ''

    def get_SpanLength(self):
        return ''

    def get_SpanLengthOdom(self):
        return ''

    def get_SpanHeight(self):
        return ''

    def get_SpanDEI(self):
        return ''

    def get_Location(self):
        return ''

    def get_Offset(self):
        return ''

    def get_DimX(self):
        return ''

    def get_DimY(self):
        return ''

    def get_DimZ(self):
        return ''

    def get_Anomlink(self):
        return ''

    def get_VideoCount(self):
        return ''

    def get_UniqueID(self):
        return ''

    def get_Dummy3(self):
        return ''

    def get_Dummy4(self):
        return ''

    def get_DummyString(self):
        return ''           


class PrismAnomaly:

    def __init__(self, src_event, navi_proj):
        self.img_dir = navi_proj.images_dir
        self.prism_imgs_dir = navi_proj.prism_imgs_dir
        self.src_event = src_event
        self.options = {
            "#01": self.get_decription,
            "#02": self.get_data_source,
            "#03": self.get_status,
            "#04": self.get_textual_assess,
            "#05": self.get_textual_second,
            "#06": self.get_report_ref_scs,
            "#07": self.get_review_date,
            "#08": self.get_apprved_by,
            "#09": self.get_approved_date,
            "#10": self.get_assessment_ref,
            "#11": self.get_assessor_name,
            "#12": self.get_imc_status,
            "#13": self.get_imc_anomaly_cat,
            "#14": self.get_kp_start_anomaly,
            "#15": self.get_date_last_mod,
        }

    def get_decription(self):
        return self.src_event['Comments (Static)']

    def get_data_source(self):
        return 'GVI'

    def get_status(self):
        return ''

    def get_textual_assess(self):
        return ''

    def get_textual_second(self):
        return ''

    def get_report_ref_scs(self):
        return ''

    def get_review_date(self):
        return ''

    def get_apprved_by(self):
        return ''

    def get_approved_date(self):
        return ''

    def get_assessment_ref(self):
        return ''

    def get_assessor_name(self):
        return ''

    def get_imc_status(self):
        return ''

    def get_imc_anomaly_cat(self):
        return ''

    def get_kp_start_anomaly(self):
        return self.src_event['KP (Dynamic)']

    def get_date_last_mod(self):
        return '20190812 214044'

    def form_opts_str(self):
        opt_strs = []
        for opt in self.opts_to_include:
            opt_strs.append(f'{opt}{self.options[opt]()}')
        
        return '\n'.join(opt_strs)

    def to_prism(self, opts_to_include):
        self.opts_to_include = [f'#{o:02}' for o in opts_to_include]
        anom_str = ','.join([
            'AreaName',
            self.src_event['pipeline_id'],
            str(self.src_event['Source (Static)']),
            self.get_image_listing(self.src_event['Source (Static)'])
        ])
        if self.opts_to_include:
            anom_str = '\n'.join([anom_str, self.form_opts_str()])

        return anom_str

    def get_image_listing(self, anom_number):
        img_paths = list(self.img_dir.glob(f'{anom_number}*.*'))
        for img_path in img_paths:
            self.prism_imgs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(img_path, self.prism_imgs_dir / img_path.name)
        imgs = [i.stem for i in img_paths]

        return ' '.join(imgs)


if __name__ == '__main__':
    
    # eff_path = Path(r'E:/NAFS/Perenco/Perenco_PRELIM/Data/Pipes/Pipe 1.eff')
    # df = NaviModelEffBurialEvents.gen_burial_events_summary(eff_path)
    # print(df[df['Burial']!='Covered'])
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(9.2, 3))
    # ax.barh(df.Burial, df.end_kp-df.start_kp, left=df.start_kp, height=0.7)

    cpp_path = Path(r'B:\MSC\_Progs\F272746_Perenco_NearshoreCP\PD272746\4_P_I\CP\FW_ Dimlington Survey Report\Dimlington Nearshore pipelines Raw Data.xlsx')
    df = pd.read_excel(cpp_path)
    (df['Potential On'] -  df['Potential Off']).plot()