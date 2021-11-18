import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

# coefs = [a,b,c] where our line is ax + by + c = 0
def line_fn(coefs):
    a,b,c = coefs
    return lambda x: (-c - a*x)/b

def draw_line(coefs,dom):
    f = line_fn(coefs)
    line_dom = [dom[0]-5,dom[1]+5]
    df = pd.DataFrame({"x":line_dom, "y": [f(x) for x in line_dom]})
    c = alt.Chart(df).mark_line(clip=True,strokeDash=[3,6],color="orange").encode(
        x = alt.X("x", scale=alt.Scale(domain=dom)),
        y = "y",
    )
    return c

st.title("Demonstration of Gradient Descent")

st.write("")

rng = np.random.default_rng()

coefs = 30*rng.random(size=(3))-20

pts = 20

df = pd.DataFrame(index = range(pts), columns= ["x","y"])

df["x"] = rng.normal(size = pts, loc = 0, scale = 5)

true_f = line_fn(coefs)

df["y"] = true_f(df["x"]) + rng.normal(size = pts, loc = 0, scale = 5)

xmin = df["x"].min()
xmax = df["x"].max()

a,b,c = coefs

st.write("Here is some data and the 'true' line underlying the data.")

st.markdown(f'''The true line is given by the equation 
                $y = {round(-a/b,2)}x {"+" if -c/b > 0 else ""} {round(-c/b,2)}.$
                ''')

chart1 = draw_line(coefs,[xmin,xmax])
chart2 = alt.Chart(df).mark_circle().encode(x = "x", y = "y")

st.altair_chart(chart1 + chart2)