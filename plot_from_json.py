import argparse
import json

from visualization.plot_results import plot_results

parser = argparse.ArgumentParser(description='Make plot from json')
parser.add_argument('json_path', type=str, help='path to json results')
parser.add_argument('out_path', type=str, help='path to save plot')
parser.add_argument('--p', type=int, default=4,
                    help='precision')
parser.add_argument('--t', type=str, default=None,
                    help='title')
parser.add_argument('--ymin', type=float, default=None,
                    help='ymin')
parser.add_argument('--ymax', type=float, default=None,
                    help='ymax')

args = parser.parse_args()

with open(args.json_path) as f:
    data = json.load(f)

max_std = max(r['std'] for search_results in data.values() for r in search_results.values())
max_score = max(r['mean'] for search_results in data.values() for r in search_results.values())
min_score = min(r['mean'] for search_results in data.values() for r in search_results.values())

plot_results(data, title=args.t, ylabel='error', save_path=args.out_path, ymax=(args.ymax or 1.02 * max_score + max_std),
                 ymin=(args.ymin or 0.98 * min_score - max_std), precision=args.p)