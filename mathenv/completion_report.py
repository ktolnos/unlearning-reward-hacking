"""Write a completion report after a math calibration job ends (including failure)."""
import argparse
import json
import subprocess
from pathlib import Path

from common import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()
    root = paths.OUT
    directory = root / f'math-calibrate-{args.job_id}'
    summary = directory / 'summary.json'
    accounting = subprocess.run(
        ['sacct', '-j', str(args.job_id), '--format=JobID,State,Elapsed,ExitCode', '-n', '-P'],
        check=True, capture_output=True, text=True).stdout.strip()
    lines = [f'# Math calibration completion: {args.job_id}', '', '## Slurm status', '',
             '```text', accounting or 'Accounting not yet available.', '```', '']
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
        except (OSError, ValueError) as exc:
            lines += [f'ERROR: Cannot read summary: {exc}', '']
        else:
            lines += [f"Probe status: {data.get('status', 'incomplete — no final status recorded')}", '',
                      '| Stage | Setting | Accuracy | Informative groups | Truncation | Pass |',
                      '|---|---|---:|---:|---:|---|']
            for row in data.get('rows', []):
                lines.append(f"| {row['stage']} | {row['label']} | {row['accuracy']:.1%} | {row['accvar']:.1%} | {row['truncation']:.1%} | {row['qualifies']} |")
            families = sorted({r['task'] for r in data.get('rows', []) if r['stage'] == 'confirm' and r['qualifies']})
            lines += ['', f'Confirmed task families in this job: {len(families)}.',
                      ', '.join(families) or 'None.', '',
                      'This report does not establish RL trainability or automatically change the training split.', '']
    else:
        lines += ['ERROR: No summary produced. Check the job log for failure before the first completed setting.', '']
    lines += [f'Summary: {summary}', f'Log: {root / f"math-calibrate-{args.job_id}.out"}', '']
    output = root / f'math-completion-{args.job_id}.md'
    temporary = output.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(output)
    print('\n'.join(lines))
    print(f'Report saved: {output}')


if __name__ == '__main__':
    main()
