from numpy.core.fromnumeric import repeat
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# st.markdown('''To do:
# * Batch size
# * Divide the coloring step from the line update step
# * Replace the grey line with the line of best fit (scikit-learn)
# * Image for contour lines?
# ''')

rng = np.random.default_rng()

pts = 20

max_updates = 2000

init_alpha = 0.01

init_batch = 1

init_theta = (4,0)

# coefs = [a,b,c] where our line is ax + by + c = 0
def line_fn(coefs):
    t0, t1 = coefs
    return lambda x: t0 + t1*x

def draw_line(coefs, dom, color="lightgrey"):
    f = line_fn(coefs)
    line_dom = [dom[0] - 5, dom[1] + 5]
    df = pd.DataFrame({"x": line_dom, "y": [f(x) for x in line_dom]})
    c = (
        alt.Chart(df)
        .mark_line(clip=True, color=color)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=dom)),
            y="y",
        )
    )
    return c

def update_theta(theta,batch,alpha):
    t0, t1 = theta
    val = 2*(t0 + t1*batch["x"] - batch["y"])/pts
    return (t0 - (alpha*val).sum(), t1 - (alpha*val*batch["x"]).sum())

def make_theta_array(df, pt_idx, alpha, batch):
    theta_arr = [init_theta]
    for i in range(max_updates):
        theta_arr.append(
                update_theta(theta_arr[i],df.loc[pt_idx[i*batch:(i+1)*batch],["x","y"]], alpha)
            )
    return theta_arr

if "batch" in st.session_state:
    batch = st.session_state["batch"]
else:
    batch = init_batch

def update():
    theta_arr = [(4,0)]
    (_, df, pt_idx) = st.session_state["data"]
    alpha = st.session_state["alpha"]
    batch = st.session_state["batch"]
    st.session_state["theta_arr"] = make_theta_array(df, pt_idx, alpha, batch)

def get_latex_for_line(m,b):
    return f'''$y = {round(m,2)}x {"+" if b >= 0 else ""} {round(b,2)}$'''

def clear_data():
    del st.session_state["data"]
    del st.session_state["step_slider"]
    del st.session_state["theta_arr"]




if "data" in st.session_state:
    (fit_coefs, df, pt_idx) = st.session_state["data"]
else:
    coefs = 30 * rng.random(size=(2)) - 20
    df = pd.DataFrame(index=range(pts), columns=["x", "y", "color"])
    df["x"] = rng.normal(size=pts, loc=0, scale=3)
    true_f = line_fn(coefs)
    df["y"] = true_f(df["x"]) + rng.normal(size=pts, loc=0, scale=7)
    df["color"] = 0
    
    reg = LinearRegression()
    reg.fit(df[["x"]], df[["y"]])
    fit_1 = reg.coef_[0][0]
    fit_0 = reg.intercept_[0]

    fit_coefs = (fit_0, fit_1)

    pt_idx = np.concatenate([rng.permutation(range(pts)) for i in range((max_updates*batch)//pts + 1)])

    st.session_state["data"] = (fit_coefs, df, pt_idx)
    

if "theta_arr" in st.session_state:
    theta_arr = st.session_state["theta_arr"]
else:
    theta_arr = [(4,0)]

    for i in range(max_updates):
        theta_arr.append(update_theta(theta_arr[i],df.loc[pt_idx[i*batch:(i+1)*batch],
                ["x","y"]], init_alpha))

    st.session_state["theta_arr"] = theta_arr


st.title("Demonstration of Gradient Descent")

st.markdown('''Some data is shown below, together with the line of best fit for that data.
There is a formula for finding that best fit line, but it can be more efficient to find 
the line instead using the iterative procedure of *gradient descent*.

Our goal is to find coefficients $t_0$, $t_1$
so that the line $y = t_1 x + t_0$ fits the data as well as possible.  We start with the guess $y = 4$ and
then gradually update the parameters.

Choices:
* The number of iterations to perform.
* The *learning rate*, which controls how much the parameters change with each update.
* The *batch size*, how many of the data points you want to consider in each step.  For stochastic
gradient descent, use a batch size of 1.
''')

xmin = df["x"].min()
xmax = df["x"].max()
ymin = df["y"].min()
ymax = df["y"].max()

chart1 = draw_line(fit_coefs, [xmin, xmax])

chart2 = alt.Chart(df).mark_circle().encode(
    x=alt.X("x",scale=alt.Scale(domain=[xmin, xmax])), 
    y=alt.Y("y",scale=alt.Scale(domain=[ymin, ymax])), 
    color=alt.Color("color:N", legend=None, scale=alt.Scale(domain=[0,1], range=["darkgrey","red"])),
    )





fit_0, fit_1 = fit_coefs


with st.sidebar:
    learn = st.slider("What learning rate?",min_value=0.0,max_value=0.2,step=0.002, value = init_alpha,
                key="alpha", on_change = update, format="%.3f")

    batch = st.slider("What batch size?",min_value=1,max_value=pts,step=1, value = init_batch,
                key="batch", on_change = update)

step = st.slider("How many updates do you want to perform?",min_value=0,max_value=max_updates,step=1,
                key = "step_slider")

df["color"] = 0

df.loc[pt_idx[step*batch:(step+1)*batch],"color"] = 1

t0, t1 = theta_arr[step]

chart1b = draw_line((t0,t1), [xmin, xmax],color="black")

c0, c1 = st.columns((7,5))

with c0:
    st.altair_chart(alt.layer(chart1,chart1b,chart2),use_container_width=True)

df_theta = pd.DataFrame(theta_arr, columns=["t0","t1"])

t0_min, t1_min = df_theta.min(axis=0) - 2
t0_max, t1_max = df_theta.max(axis=0) + 2

chart_theta = alt.Chart(df_theta.loc[:step]).mark_circle().encode(
    x=alt.X("t0",scale=alt.Scale(domain=[t0_min, t0_max])), 
    y=alt.Y("t1",scale=alt.Scale(domain=[t1_min, t1_max])), 
)

chart_fit = alt.Chart(pd.DataFrame({"t0":[fit_0],"t1":[fit_1]})).mark_point(
    color="red",
    shape="diamond",
    size=100,
    ).encode(
        x = "t0",
        y = "t1",
    )

with c1:
    st.altair_chart(chart_theta + chart_fit,use_container_width=True)

st.button("Get new data",on_click=clear_data)

st.markdown(
    f"""The line of best fit is given by the equation 
        {get_latex_for_line(fit_1, fit_0)}.  (Shown in grey.)
                """
)

st.markdown(f"Our current guess for the line is given by {get_latex_for_line(t1, t0)}.  (Shown in black.)")

st.write('''The chart on the right shows the estimated coefficients so far (in blue) and the 
best-fit coefficients (in red).
''')


