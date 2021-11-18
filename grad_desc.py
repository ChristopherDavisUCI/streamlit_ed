from numpy.core.fromnumeric import repeat
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

st.markdown('''To do:
* Batch size
* Divide the coloring step from the line update step
* Replace the grey line with the line of best fit (scikit-learn)
''')

rng = np.random.default_rng()

pts = 20

max_updates = 200

init_alpha = 0.01

# coefs = [a,b,c] where our line is ax + by + c = 0
def line_fn(coefs):
    a, b, c = coefs
    return lambda x: (-c - a * x) / b

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

def update_theta(theta,pt,alpha):
    t0, t1 = theta
    x, y = pt
    val = 2*(t0 + t1*x - y)/pts
    return (t0 - alpha*val, t1 - alpha*val*x)

# Remove the repetition from this and the below code
def update_alpha():
    theta_arr = [(4,0)]

    for i in range(max_updates):
        theta_arr.append(update_theta(theta_arr[i],df.loc[pt_idx[i],["x","y"]],
            st.session_state["alpha"]))

    st.session_state["theta_arr"] = theta_arr

def get_latex_for_line(m,b):
    return f'''$y = {round(m,2)}x {"+" if b >= 0 else ""} {round(b,2)}$'''

def clear_data():
    del st.session_state["data"]
    del st.session_state["step_slider"]
    del st.session_state["theta_arr"]




if "data" in st.session_state:
    (coefs, df, pt_idx) = st.session_state["data"]
else:
    coefs = 30 * rng.random(size=(3)) - 20
    df = pd.DataFrame(index=range(pts), columns=["x", "y", "color"])
    df["x"] = rng.normal(size=pts, loc=0, scale=3)
    true_f = line_fn(coefs)
    df["y"] = true_f(df["x"]) + rng.normal(size=pts, loc=0, scale=5)
    df["color"] = 0
    
    pt_idx = np.concatenate([rng.permutation(range(pts)) for i in range(max_updates//pts + 1)])

    st.session_state["data"] = (coefs, df, pt_idx)
    

if "theta_arr" in st.session_state:
    theta_arr = st.session_state["theta_arr"]
else:
    theta_arr = [(4,0)]

    for i in range(max_updates):
        theta_arr.append(update_theta(theta_arr[i],df.loc[pt_idx[i],["x","y"]], init_alpha))

    st.session_state["theta_arr"] = theta_arr


st.title("Demonstration of Gradient Descent")

xmin = df["x"].min()
xmax = df["x"].max()
ymin = df["y"].min()
ymax = df["y"].max()

a, b, c = coefs

chart1 = draw_line(coefs, [xmin, xmax])

chart2 = alt.Chart(df).mark_circle().encode(
    x=alt.X("x",scale=alt.Scale(domain=[xmin, xmax])), 
    y=alt.Y("y",scale=alt.Scale(domain=[ymin, ymax])), 
    color=alt.Color("color:N", scale=alt.Scale(domain=[0,1], range=["darkgrey","red"])),
    )

learn = st.slider("What learning rate?",min_value=0.0,max_value=0.2,step=0.002, value = init_alpha,
            key="alpha", on_change = update_alpha)

step = st.slider("How many updates do you want to perform?",min_value=0,max_value=max_updates,step=1,
                key = "step_slider")

df["color"] = 0

df.loc[pt_idx[step],"color"] = 1

t0, t1 = theta_arr[step]

st.markdown(
    f"""The true line is given by the equation 
        {get_latex_for_line(-a/b, -c/b)}.  (Shown in grey.)
                """
)

st.markdown(f"Our current guess for the line is given by {get_latex_for_line(t1, t0)}.  (Shown in black.)")

chart1b = draw_line((-t1,1,-t0), [xmin, xmax],color="black")

st.altair_chart(alt.layer(chart1,chart1b,chart2))


st.button("Get new data",on_click=clear_data)
