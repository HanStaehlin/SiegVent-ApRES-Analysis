path = 'proc/apres/visualization_app.py'
with open(path, 'r') as f:
    text = f.read()

text = text.replace('np.clip(echogram_db, -25, 50)', 'np.clip(echogram_db, -10, 50)')
text = text.replace('cmin = -25', 'cmin = -10')
text = text.replace('np.clip(echo_db_local, -25, 50)', 'np.clip(echo_db_local, -10, 50)')
text = text.replace('np.clip(10 * np.log10(echo_sel**2 + 1e-30), -25, 50)', 'np.clip(10 * np.log10(echo_sel**2 + 1e-30), -10, 50)')

with open(path, 'w') as f:
    f.write(text)
print("done")
