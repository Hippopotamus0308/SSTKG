# STKG
knowledge graph formation using spatial-temporal data

# graph version no.1
## Graph formation
### ALL ENTITIES ARE "STORES" POI
This will make the graph consistent while not homogeneous. won't set things like times/different categories as entity type(will make things worse, high chance of overfit)

#### on the other hands, set categories as an attribute
definitely, in an area sellings get influenced by 'vicious/healthy competition' or 'reciprocal symbiosis', stores are related by categories

### RELATION = F(distance, categories, time)
in the graph, just one kinds of relation--influence, which is vectorized numerical, not categorical.

For example, (Macdonald Randwick)--\[influence\]-->(UNSW Pho house) means some influence the Mac will put on pho house.

Set it to be vectorized, considering selling contains online & offline. 

can be expanded: In current version it's n*3, while the 3 refers to:

- quite time (normally weekday)

- busy time (normally weekend + Friday)

- non-overlap time (in case some stores open while others close)

Benifits: Manhattan Distance cannot be easily changed in a short period of time(especially we can only get the latitude and longitude), and in relatively close area like one/two blocks, the relation ship won't change rapidly in a short time, thus make the relation stable.

### What is the ENTITIY?
not just a 'brand + id', we manage to store time into it.

#### entity structure, thankfully using document structure:
- placekey
- Brand+ID(by safe graph)
- top category
- sub category
- \[latitude, longitude\]
- open hours
- spend records
 - weekly records
 - daily records

### What is the FACT?
Fact can be seen as entity * relation * entity * time, past influence records are put in the fact, while using these some further prediction can be also put into fact.

#### Why use it?
To seperately store fact, it's easy to get the data out and use for embedding while we can freedomly change data in entity/relation without worrying about influence past records.

## Verification:
1. (?, r, t, τ) Use past month’s records to predict this month’s records
2. (h, ?, t, τ)relation prediction: predicting certain influence value next month
3. (h, r, t, ?)time prediction: predicting under certain relation will it be quite/busy time for store, or weekday/weekends/holiday

## Current Problem
### Missing data (spend data)
brand_info is not complete(or problem in disjoining dataframe)

### Knowledge graph
Reducing size (wipe out |influence| < threshold in subgraph)
? Time period settings (is it possible that past records in other store is related to later records in another store?)
