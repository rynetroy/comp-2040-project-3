![Downtown Winnipeg - Portage Place construction, January 2026](visuals/header.png)


# Reimagining Downtown Winnipeg: A Multi-Billion Dollar Transition From Retail to Residential (2010–2026)

**Troy Dela Rosa**

![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)

**Downtown Winnipeg isn't dying. It's being rebuilt into something different, and the results may not show up until 2027 or 2028.**

This analysis is intended for city planners, developers, local business groups, and civic stakeholders evaluating downtown redevelopment strategy.

Major projects, including the Portage Place redevelopment and the former Bay building conversion, are driving a broader pipeline of more than $2B in planned, committed, and phased downtown investment. Not all of that investment is actively under construction today. Estimates are compiled from public announcements and may include multi-phase projects.

More than 1,600 new apartments are confirmed, with additional units proposed. Office vacancy has doubled. Retail is down 63% from its 2013 peak. These are not contradictory signals. They are the same story at different points in time.

The cranes are real. The empty storefronts are real. The question is whether the planned pipeline turns into completed housing, foot traffic, and street-level recovery.

**In short:** Downtown Winnipeg is in the messy middle of a multi-billion dollar transformation that may not be fully visible on the street for another two years.

---

## Business Question

Is downtown Winnipeg in decline, or is it moving through a delayed transition from office and retail dependence toward residential, institutional, and mixed-use activity?

---

## Executive Snapshot

| Metric | Finding |
|---|---|
| Redevelopment pipeline | $2B+ planned, committed, and phased investment |
| New apartments confirmed or proposed | 1,635-1,793 units |
| Office space sitting empty | 18.6% in 2025 |
| Retail activity vs. 2013 | Down ~63% |
| Downtown Health Score | 63 / 100, constructed scenario index, mixed performance, not structural decline |

The **Downtown Health Score** is a constructed scenario index combining business activity, residential growth, investment pipeline strength, spatial concentration, and office vacancy pressure. It is not an official city metric. It is used here as a directional way to compare whether downtown is improving, stagnating, or worsening.

---

## Office Vacancy Has Nearly Doubled, But the Sharpest Increase Has Slowed

![Office vacancy rates: Winnipeg vs. the national average, 2016-2025](visuals/fig_vacancy.png)

Vacancy rose from 8.7% in 2016 to 18.6% in 2025. However, the pace slowed sharply in 2024-2025, rising just 0.3 percentage points across both years. The sharpest phase of the increase appears to have slowed.

---

## Restaurants Are Recovering. Retail Is Not.

![Downtown business activity by sector, 2010-2024 compared to 2013 levels](visuals/fig_business_activity.png)

Retail activity is down about 63% from its 2013 peak, which helps explain the visible empty storefronts. Food and service activity has recovered faster than retail since COVID. The data supports the visible recovery in downtown food, cafe, and service activity.

---

## The Construction Is Concentrated in Six City Blocks

![Map of construction and development projects across downtown Winnipeg](visuals/fig_spatial.png)

Most major redevelopment activity is concentrated along Portage Avenue between the former Bay building and True North Square. Walk east into the Exchange District and the picture looks different. Downtown is not behaving as one uniform place. It is splitting into corridors with different levels of investment, activity, and transition pressure.

---

## It All Depends on Whether the Buildings Get Finished

![Four scenarios for downtown depending on construction timelines](visuals/fig_scenarios.png)

If projects finish on time and people move in, the Downtown Health Score could reach **73 out of 100**. If construction is delayed by two or more years, it could fall to **44**.

The empty storefronts are unlikely to fill back up before the new residential base arrives. That means the transition may not become visible at street level until 2027 or 2028.

---

## Implications

- Prioritize completion of major residential projects so new population density reaches street-level businesses sooner.
- Support interim retail activation through pop-ups, short-term leases, and small business incentives while construction is still ongoing.
- Concentrate public realm, safety, and streetscape improvements along the Portage Avenue redevelopment corridor where investment is already clustered.
- Track vacancy, business openings, and residential occupancy together rather than treating each indicator separately.
- Treat downtown as a transition system, not a single success-or-failure story.

---

## Final Takeaway

Downtown is not failing. It is lagging its own transformation timeline.

This analysis builds on a growing local conversation about downtown Winnipeg's future. Recent reporting has increasingly framed major projects like Portage Place and the former Bay building not as a return to the old retail model, but as part of a broader shift toward housing, institutional uses, and adaptive reuse.

This project adds a data-based framework for connecting those redevelopment stories to vacancy trends, business activity, spatial concentration, and delivery timing.

---

## About the Data

Vacancy figures are based on CBRE market reports, with 4 of 10 years estimated. Business activity before 2021 was reconstructed from supporting sources and should be treated as directional rather than exact. Investment amounts were compiled from press releases and public reporting.

Full source registry: [`data/downtown_wpg_sources_2026.csv`](data/downtown_wpg_sources_2026.csv)

**To run:** Open `notebooks/stakeholder_report.ipynb`
