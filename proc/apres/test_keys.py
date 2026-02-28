import pickle
import sys

try:
    with open('/Users/hannesstahlin/SiegVent2023-Geology/proc/apres/layer_analysis_results.pkl', 'rb') as f:
        res = pickle.load(f)
    print("Keys in results:", res.keys())
    if 'velocity' in res:
        print("Keys in velocity:", res['velocity'].keys())
except Exception as e:
    print(e)
