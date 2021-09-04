# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Productivity
#
# As I've gotten older, I've realized what makes me producitve is not more hours
#
# _The irony that I've written this entire post using complex technology is not lost on me. This post was to satisfy my own 'technological itch'_

# %pylab inline
import matplotlib.pyplot as plt
import numpy as np

#

# +
plt.xkcd()
height_in_inches = 12
mpl.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fig1, (
    (ax_thought_more_productive, ax_normal_hours),
    (ax_actually, ax_more_hours),
) = plt.subplots(2, 2)

ax_thought_more_productive.set_title(" What I thought would make me \n more productive")
ax_thought_more_productive.pie(
    [100, 10], labels=["More hours", "More Money"], startangle=200
)
ax_thought_more_productive.axis(
    "equal"
)  # Equal aspect ratio ensures that pie is drawn as a circle.

labels = ["More hours", "Exercise", "Healthy Eating", "Sleep", "Time Off"]
sizes = [30, 10, 15, 30, 10]
explode = (0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax_actually.pie(sizes, explode=explode, labels=labels, startangle=300)
ax_actually.set_title("What Actually Does")
ax_actually.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

ax_normal_hours.set_title("Time distribution working normal hours")
ax_normal_hours.barh(y=[""], width=[44], color="red")
ax_normal_hours.barh(y=[""], width=[36], color="royalblue")
ax_normal_hours.legend(["Dicking Around", "Working"])
ax_normal_hours.set_xlim(0, 70)
ax_more_hours.set_title("Time distribution working more hours")
ax_more_hours.barh(y=[""], width=[64], color="red")
ax_more_hours.barh(y=[""], width=[44], color="royalblue")
ax_more_hours.legend(["Dicking Around", "Working"])
ax_more_hours.set_xlim(0, 70)


# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#  plt.subplots_adjust(hspace=1)k


plt.show()
# -



# +
plt.xkcd()
height_in_inches = 8
import matplotlib.patches as mpatches


mpl.rc("figure", figsize=(1 * height_in_inches, height_in_inches))

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fig1, (
    (ax_problem_size)
) = plt.subplots(1, 1)


ax_problem_size.set_title("How big a problem feels")
ax_problem_size.bar(x=[""], height=[0], color="red")
ax_problem_size.bar(x=["At Night"], height=[40], color="red")
ax_problem_size.bar(x=["Next Morning"], height=[7], color="blue")
ax_problem_size.set_xlim(0.5,2.5)

night  = mpatches.Patch(color='red', label='At Night')
day = mpatches.Patch(color='Blue', label='Next Morning')

ax_problem_size.legend(handles=[night,day])



plt.show()

# +
plt.xkcd()
height_in_inches = 12
mpl.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fig1,  (career_pie, life_pie) = plt.subplots(1, 2)

career_pie.axis(
    "equal"
)  # Equal aspect ratio ensures that pie is drawn as a circle.

labels = ["Job Title", "Compensation"]
sizes = [40, 60]
explode = [0.1,0]
career_pie.pie(sizes, explode=explode, labels=labels, startangle=0)
career_pie.set_title("Career Discussions\nDeafult")
career_pie.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

labels = ["Job Title", "Compensation", "Health", "Hobbies", "Friends and Family"]
sizes = [20, 15, 10, 10, 10]
explode = (0.0, 0, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
life_pie.set_title("Career Discussions\nWith the big picture")
life_pie.pie(sizes, explode=explode, labels=labels, startangle=300)

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
#  plt.subplots_adjust(hspace=1)k


plt.show()
# -


