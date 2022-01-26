from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = '/home/masterbin-iiau/Desktop/tracking_datasets/LaSOT'
    settings.network_path = '/home/yangjinyu/rgbd_tracker/CLGSD/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = '/home/yangjinyu/rgbd_tracker/CLGSD/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''

    return settings

