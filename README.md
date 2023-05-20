# STKG
knowledge graph formation using spatial-temporal data

# graph version no.1
## Graph formation
### ALL ENTITIES ARE "STORES" POI
This will make the graph homogenious. won't set things like times/different categories as entity type(will make things worse, high chance of overfit)

#### on the other hands, set categories as an attribute
definitely, in an area sellings get influenced by 'vicious/healthy competition' or 'reciprocal symbiosis', stores are related by categories

### RELATION = F(distance, categories)
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
Brand+ID(by safe graph)
- top category
- sub category
- \[latitude, longitude\]
- open hours
- spend records
- - weekly records
- - daily records
   