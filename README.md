# tomGame

Javascript port of ([http://mbb-team.github.io/VBA-toolbox/wiki/ktom/]()).

#ToM.js

This is the engine of the project (the html files are merely implementations). ToM is invoked as follows:

```javascript
tom(tomLevel,ROLES.ROLE_HIDER) // set up the TOM as the hider
answers[roundNumber++] = {hider: tom.getNextAnswer(), seeker: Math.round(Math.random())] // Get the players' answers for the round
tom.processFeedback({self: answers[roundNumber-1].hider, opponent: answers[roundNumber-1].seeker}) // get TOM to update its beliefs based on the evidence
```
