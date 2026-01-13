def apply_fitpulse_plotly_theme(fig, height=450):
    fig.update_layout(
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#E5E7EB"),

        legend=dict(
            font=dict(
                color="#F8FAFC",
                size=13
            ),
            bgcolor="rgba(2,6,23,0.6)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1
        ),

        height=height
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.2)"
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.2)"
    )

    return fig
