# exploratory data analysis with SeaBorn
import seaborn as sns

sns.factorplot('income', 'capital-gain', hue='sex', data=data, kind='bar', col='race', row='relationship')