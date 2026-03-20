from pathlib import Path
import pandas as pd

from pypipe.events import NaviModelPipelineProject
 

if __name__ == '__main__':

    # retreive NaviModel project paths from project listing
    proj_paths_path = Path(r'Perenco\pipeline_project_paths.csv')
    proj_paths = pd.read_csv(proj_paths_path, index_col='Pipeline')
    prism_dir = Path(r'Perenco\PRISM')

    # for each NaviModel project, export events and anomalies
    for pipeline, meta in proj_paths.iterrows():
        pipeline_proj = NaviModelPipelineProject(pipeline, meta)
        events_path = pipeline_proj.export_events()
        anomalies_path = pipeline_proj.export_anomalies()

