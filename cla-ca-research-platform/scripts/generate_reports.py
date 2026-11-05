import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cla_ca.reporting import generate_reports_and_figures

if __name__ == '__main__':
    print("Generating reporting figures and summaries...")
    generate_reports_and_figures()
