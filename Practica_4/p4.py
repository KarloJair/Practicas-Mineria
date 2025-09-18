import pandas as pd
import scipy.stats as stats

df = pd.read_csv("vgsales.csv")

var_dep = "Global_Sales"
var_cat = "Genre"

groups = [group[var_dep].values for name, group in df.groupby(var_cat)]

anova = stats.f_oneway(*groups)
print(anova)

# Tests en grupos específicos

action_sales = df[df["Genre"]=="Action"]["Global_Sales"]
sports_sales = df[df["Genre"]=="Sports"]["Global_Sales"]

t_test = stats.ttest_ind(action_sales, sports_sales, equal_var=False)
print(t_test)

racing_sales = df[df["Genre"]=="Racing"]["Global_Sales"]
plataform_sales = df[df["Genre"]=="Platform"]["Global_Sales"]

t_test = stats.ttest_ind(racing_sales, plataform_sales, equal_var=False)
print(t_test)