import csv

with open('submission.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Sort: primary = score descending, secondary = candidate_id ascending
rows.sort(key=lambda r: (-float(r['score']), r['candidate_id']))

# Re-assign ranks
for i, row in enumerate(rows):
    row['rank'] = i + 1

# Write back
with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
    writer.writeheader()
    writer.writerows(rows)

print("Done! Run validate_submission.py again.")