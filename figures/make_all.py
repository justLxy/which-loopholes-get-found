"""Rebuild every paper figure from the committed results: python -m figures.make_all

Every figure needs only matplotlib. Figure 1 (figures/gen_fig_summary.py) replaces the earlier takeover figure and
the TikZ overview.
"""

from figures import gen_fig_auroc, gen_fig_fpr_heatmap, gen_fig_h1, gen_fig_h2_h3, gen_fig_prospective, gen_fig_summary

if __name__ == "__main__":
    for module in (gen_fig_summary, gen_fig_fpr_heatmap, gen_fig_h1, gen_fig_auroc, gen_fig_prospective,
                   gen_fig_h2_h3):
        module.main()
