# Data sources

| Layer | Source | Access in pipeline | Notes |
|---|---|---|---|
| Rainfall | CHIRPS Daily | Google Earth Engine | Seasonal and annual rainfall totals |
| Vegetation | MODIS MOD13Q1 | Google Earth Engine | NDVI/EVI seasonal means |
| Nightlights | NASA VIIRS Black Marble VNP46A2 | Google Earth Engine | Capacity/economic-activity proxy, available from 2012 |
| Population | WorldPop | Google Earth Engine | District population/exposure aggregation |
| Macro indicators | World Bank WDI | World Bank API | GDP, agriculture, electricity, population, land indicators |
| Boundaries | geoBoundaries by default, optional COD-AB | API/local file | Use one fixed boundary reference |
| Agriculture validation | FAOSTAT | User-supplied CSV currently | Use according to chosen crop/item/element |
| Survey validation | LSMS/UNPS/DHS | User-supplied controlled-access files | Do not redistribute raw microdata |
