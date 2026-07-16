"""Assemble the self-contained RH Smith Chart HTML from template + data."""
import json

with open("data/chart_data_web.json") as f:
    DATA = f.read()

with open("chart_template.html") as f:
    tpl = f.read()

html = tpl.replace("/*__DATA__*/", "const DATA = " + DATA + ";")
with open("rh_smith_chart.html", "w") as f:
    f.write(html)
print("bytes:", len(html))
