# tomGame

Javascript port of ([http://mbb-team.github.io/VBA-toolbox/wiki/ktom/]()).

#ToM.js

This is the engine of the project (the html files are merely implementations). ToM is invoked as follows:

```javascript
// set up the TOM as the hider
tom(tomLevel,ROLES.ROLE_HIDER);
// Get the players' answers for the round
answers[roundNumber++] = {hider: tom.getNextAnswer(), seeker: Math.round(Math.random())];
// Format the answers appropriately for ToM
tomAnswers = {self: answers[roundNumber-1].hider, opponent: answers[roundNumber-1].seeker};
// get TOM to update its beliefs based on the evidence
tom.processFeedback(tomAnswers);
```
