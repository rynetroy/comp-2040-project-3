#  Downtown Winnipeg Business Decline Analysis  
## comp-2040-project-3

##  Project Overview  
This project analyzes business license data from the City of Winnipeg to explore trends in downtown business activity. The goal is to identify whether downtown Winnipeg is experiencing a decline in active businesses using license status as a proxy for business closure.

The analysis focuses on uncovering patterns across time, industry, and location to better understand shifts in the downtown economic landscape.

---

##  Objectives  
- Analyze trends in active vs inactive businesses over time  
- Identify industries most affected by business closures  
- Explore geographic patterns within downtown Winnipeg  
- Apply a simple predictive model to estimate business closure likelihood  

---

## 📊 Dataset  
- **Source:** City of Winnipeg Open Data  
- **Dataset:** Business Licenses  
- **Records:** 7,300+ business licenses  

**Key Fields:**  
- `Status` (Active, Expired, Cancelled)  
- `Issue Date`, `Expiry Date`  
- `Neighbourhood Name`  
- `Subdescription` (business type)  

---

##  Methodology  

### Data Preparation  
- Cleaned column names for consistency  
- Converted date fields to datetime format  
- Removed records with missing key location data  

### Defining Business Closure  
The dataset does not explicitly indicate business closures.  
To address this, businesses with a status of **“Expired” or “Cancelled”** were classified as *closed*.  

### Downtown Scope  
Downtown Winnipeg was defined using the following neighbourhoods:  
- Downtown  
- Exchange District  
- Central Park  
- South Portage  
- Broadway-Assiniboine  

### Exploratory Data Analysis  
- Time-series analysis of closures  
- Industry-level breakdown of inactive businesses  
- Comparison of active vs closed businesses  

### Predictive Modeling  
- Model: Decision Tree Classifier  
- Target: `is_closed`  
- Features: Year (derived from issue date)  
- Goal: Estimate likelihood of business closure  
