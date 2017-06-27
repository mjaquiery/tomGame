/* Script implements variational bayes equations from Devaine et al. 2014
    Code: Matt Jaquiery (2017)

    Usage:
    >tom(tomLevel,ROLES.ROLE_HIDER) // set up the TOM as the hider
    >answers[roundNumber++] = [tom.getNextAnswer(), Math.round(Math.random())] // Get the players' answers for the round
    >tom.processFeedback(answers[roundNumber-1]) // get TOM to update its beliefs based on the evidence
*/
var ROLE_NAMES = {0: "HIDER", 1: "SEEKER"};
var ROLES = {HIDER: 0, SEEKER: 1};
var TOM_TYPES = {NON_TOM: 0, TOM: 1};

var payoffMatrix =  [
                        [[0,1],[1,0]],
                        [[1,0],[0,1]]
                    ]; // h= hider = 0, s=seeker=1. Usage: payoffMatrix[hiderAnswer][seekerAnswer][role]

/* ToM is an agent of level 0 or 1 which attempts to model the behaviour of an opponent with either a random bias (0) or a level 0 ToM. The original version is significantly more sophisticated but level 1 is the maximum needed here. Fuller details can be found in the matlab VBA toolbox.
VBA: http://mbb-team.github.io/VBA-toolbox/
Wiki: http://mbb-team.github.io/VBA-toolbox/wiki/ktom/

This initiator function performs the logic of prepare_kToM.m
*/
function tom(level,role,debug) {
    this.type = TOM_TYPES.TOM;
    this.debug = (debug==true);
    this.showDebugInfo = showDebugInfo;

    this.getNextAnswer = getNextAnswer;
    this.processFeedback = processFeedback;
    this.evolveHiddenStates = evolveHiddenStates;
    this.getAnswerProbability = getAnswerProbability;
    this.getIncentive = getIncentive;
    // options.inF (prepare_kToM.m)
    this.role = (role==1)? 1 : 0; // inF.player
    this.level = level; // inF.lev
    this.game = payoffMatrix; // inF.game
    // inF.diluteP not used; we ignore forgetting
    this.evolvingParameterIndex = 0; // inF.indParev
    this.evolvingParameters = [1,0,0]; // inF.dummyPar
    this.observationParameterIndex = 1; // inF.indParobs
    // End of inF parameters.
    // options.inG parameters include all inF parameters and also a description of the opponent model and a total parameter count.
    this.totalParameterIndex = this.evolvingParameterIndex+this.observationParameterIndex; // inF.npara
    this.opponentModel = []; // inF.indlev
    // Configue hidden states indexing if we're not 0ToM
    if(this.level==1) { // We only ever deal with level 0 and level 1. We don't need to worry about tracking the relative predictive capability of multiple levels of ToM.
        this.opponentModel = {
            // These values are INDEXES used for referencing hiddenStates
            level: 0, // k - Opponent's sophistication
            hiddenStates: [0,1], // x - Opponent's hidden states; 0ToM has 2 hidden states: E[log-odds] and V[log-odds]
            f: 2, // f - x(theta) = log odds of P(o=1)
            df: [3,4,5], // df - delta-F: d(x)/d(theta)
            Par: [6,7,8,9,10,11] // Par - E[theta] and V[theta]; Bayes stuff
        };
    }
    // End of inG properties
    // Non-structural properties:
    this.hiddenStates = []; // x - hidden state information - 1ToM has 12 variables containing info indexed by this.opponentModel
    this.initialPrior = this.evolveHiddenStates(-1); // options.priors.muX0
    this.hiddenStates[0] = this.evolveHiddenStates(0); // Set up the first level of hidden states
    this.theta = -1*Math.log(2); // theta - prior volatility
    this.phi = [-2,0,1]; // temperature, bias, perseveration
}

function getNextAnswer(processingOther) {
    return (Math.random()<this.getAnswerProbability(processingOther))? 1 : 0;
}

// MatLab version g_kToM.m and ObsRecGen.m
function getAnswerProbability(processingOther, overrideVars) {
    var level = (processingOther)? this.opponentModel.level : this.level;
    var role = (processingOther)? 1-this.role : this.role;
    var states = (processingOther)? [this.hiddenStates[this.hiddenStates.length-1][this.opponentModel.hiddenStates[0]],this.hiddenStates[this.hiddenStates.length-1][this.opponentModel.hiddenStates[1]]] : this.hiddenStates[this.hiddenStates.length-1]; // hidden states is the last set of hidden states for the agent or its internal simulation (internal is always 0Tom)
    var beta = this.phi[0];
    if(typeof overrideVars=="object") {
        if(typeof overrideVars.hiddenStates != "undefined")
            states = overrideVars.hiddenStates;
        if(typeof overrideVars.beta != "undefined")
            beta = overrideVars.beta;
    }
    var pOP; // Opponent's probability of picking 1
    var Vx;
    var a = 0.36; // ObsRecGen.m line 28
    if(level==0) {
        var Ex = states[0]; // E[log-odds of P(opponentChoice=1)]
        Vx = Math.exp(states[1]); // V[log-odds of P(opponentChoice=1)]
        pOP = sigmoid(Ex / Math.sqrt(1+a*Vx)); // P(opponentChoice=1)
    } else {
        var pK = 1; // assume opponent is 0ToM
        var f = states[this.opponentModel.f];
        Vx = 0;
        for (var i=0;i<this.opponentModel.df.length;i++) {
            var df = states[this.opponentModel.df[i]];
            var sigma = Math.exp(states[this.opponentModel.Par[1+2*i]]); // V[theta] QUESTION: should this be .Par[2*i]? ObsRecGen.m line 64
            Vx = Vx+(sigma*Math.pow(df,2)); // V[x(theta)]
        }
        pOP = sigmoid(f/Math.sqrt(1+a*Vx)); // P(opponentChoice=1)
    }
    // Now we know pOP we can work out our own answer
    var bias = this.phi[1];
    var incentive = this.getIncentive(pOP,Math.exp(beta),role,this.game);
    return sigmoid(incentive+bias);
}

// MatLab fplayer.m
// Player incentive function
function getIncentive(pOP, beta, role, game) {
    var otherRole = (this.role==0)? 1 : 0;
    var decisionVariable;
    var c0;
    var c1;
    if(role == ROLES.HIDER) { // NOTE: got confused here; may have to reverse this to SEEKER
        c0 = pOP*(game[0][0][role]-game[0][1][role]);
        c1 = (1-pOP)*(game[1][0][role]-game[1][1][role]);
    } else {
        c0 = pOP*(game[0][0][role]-game[1][0][role]);
        c1 = (1-pOP)*(game[0][1][role]-game[1][1][role]);
    }
    decisionVariable = (c0+c1)/beta;
    return decisionVariable;
}

// Update our hidden states based on answers
function processFeedback(answers,processingOther) {
    this.hiddenStates[this.hiddenStates.length] = this.evolveHiddenStates(answers,processingOther);
    if(this.debug)
        console.log(this.hiddenStates[this.hiddenStates.length-1]);
}
    // MatLab version f_kToM.m and RecToMfunction.m
    // answers is either -1|0 (special cases for initialisation) or
    // answers = {opponent: 0|1, choice: 0|1}
    // If we're processingOther answers is automatically reversed.
function evolveHiddenStates(answers,processingOther,overrideVars) {
    var prior;
    var posterior;
    var level = (processingOther)? this.opponentModel.level : this.level;
    answers.opponent = (processingOther)? answers.self : answers.opponent; // Flip answers if necessary (we only need opponent answer for 0ToM, and 1ToM can't be processingOther'd)
    var theta = this.theta;
    var hiddenStates = (processingOther)? [this.hiddenStates[this.hiddenStates.length-1][this.opponentModel.hiddenStates[0]],this.hiddenStates[this.hiddenStates.length-1][this.opponentModel.hiddenStates[1]]] : this.hiddenStates[this.hiddenStates.length-1]; // hidden states is the last set of hidden states for the agent or its internal simulation
    if(typeof overrideVars=="object") {
        if (typeof overrideVars.theta != "undefined")
            theta = overrideVars.theta;
        if(typeof overrideVars.hiddenStates != "undefined")
            hiddenStates = overrideVars.hiddenStates;
    }
    // Special cases
    // [these are only ever invoked by 1ToM]
    if(answers===-1) { // setting initial prior, see prepare_kToM.m [lines 47-63]
        if(level==0)
            return [0,0]; // 0ToM has flat priors
        prior = [];
        for(var i=0;i<12;i++)
            prior[i] = 0; // Zero-fill initial prior
        posterior = prior;
        // Tweak some values
        posterior[this.opponentModel.Par[1]] = -1; // Set opponent learning volatility to -1 (efficient learning)
        posterior[this.opponentModel.Par[2*this.evolvingParameterIndex+1]] = -1; // E(log(exploration temperature))
        return posterior;
    } else if (answers==0) { // settting first hidden state based on initial prior
        return this.initialPrior; // just get the initialPrior value and use that
    }
    prior = hiddenStates; // prior is the last set of hidden states (i.e. the previous round's posterior)

    if(typeof answers!="object")
        return prior; // If we didn't get answers (e.g. no user response) then don't update hidden states

    // now into RecToMfunction.m
    if(level == 0) {
        // 0ToM case, MatLab function evolution0bisND
        var E0 = prior[0]; // current E[log-odds]
        var V0 = Math.exp(prior[1]); // current V[log-odds]
        var p0 = sigmoid(E0); // current estimate of P(opponentChoice=1)
        var volatility = Math.exp(theta);
        var V = 1 / ((1/(volatility+V0)) + p0*(1-p0)); // updated V[log-odds]
        var E = E0 + V*(answers.opponent-p0); // updated E[log-odds] via Laplace-Kalman update rule
        return [inverseSigmoid(sigmoid(E)),Math.log(V)]; // Packaged for numerical reasons
    }
    // 1ToM case, back to RecToMfunction.m
    // NOTE: we skip P(k') because we assume k'=0
    posterior = prior;
    // Update E[theta] and V[theta]
    var pK = 1; // Probability opponent level = 0 see note above
    var pOP = sigmoid(prior[this.opponentModel.f]); // P(opponentAnswer=1)
    for (var i=0;i<this.opponentModel.df.length;i++) { // Update all the hidden states tracking opponent parameters
        var df = prior[this.opponentModel.df[i]]; // d[x(theta)]/dtheta
        var V0 = Math.exp(prior[this.opponentModel.Par[1+2*i]]) + Math.exp(theta)*prior[this.evolvingParameters[i]]; // diluted prior variance
        var Vu = 1 / ((1/V0)+pK*pOP*(1-pOP)*Math.pow(df,2)); // posterior variance
        var E0 = prior[this.opponentModel.Par[2*i]]; // Prior mean
        var Eu = E0 + pK*Vu*(answers.opponent-pOP)*df; // Posterior mean
        // store values with numerical tweaks
        posterior[this.opponentModel.Par[1+2*i]] = Math.log(Vu);
        posterior[this.opponentModel.Par[2*i]] = inverseSigmoid(sigmoid(Eu));
    }
    // Technically that's all we need (formally). But to keep par with the MatLab we augment the hidden states with dummy states derived from the sufficient statistics.
    // Simulate opponent's hidden states
    var opponentHiddenStates = this.evolveHiddenStates(answers,true);
    posterior[this.opponentModel.hiddenStates[0]] = opponentHiddenStates[0];
    posterior[this.opponentModel.hiddenStates[1]] = opponentHiddenStates[1];
    // Simulate opponent's f
    posterior[this.opponentModel.f] = inverseSigmoid(this.getAnswerProbability(true));
    // Get derivatives df
    var dfdP = [];
    for (var i=0;i<this.evolvingParameterIndex;i++) { // Loop over all evolve params
        var dP = Math.exp(-4)*prior[this.opponentModel.Par[2*i]]; // small parameter increment
        if(Math.abs(dP)<Math.exp(-4))
            dP = Math.exp(-4);
        var temp = {};
        temp.theta = prior[this.opponentModel.Par[2*i]]+dP // small increase in parameter 1
        temp.hiddenStates = prior; // QUESTION: should we use current best estimate (posterior) rather than prior here?
        var Xpdp = this.evolveHiddenStates(answers, true, temp);
        var fpdp = inverseSigmoid(this.getAnswerProbability(true,{hiddenStates: Xpdp}));
        var dfdP = (fpdp - posterior[this.opponentModel.f]) / dP; // Gradient w.r.t param 1
        posterior[this.opponentModel.df[i]] = dfdP;
    }
    // Derive fx wrt observational parameters
    dfdP = [];
    for (var i=0;i<this.observationParameterIndex;i++) {
        var dP = Math.exp(-4)*prior[this.opponentModel.Par[(2*this.evolvingParameterIndex)+2*i]]; // small parameter increment
        if(Math.abs(dP)<Math.exp(-4))
            dP = Math.exp(-4);
        var temp = {};
        temp.hiddenStates = prior;
        temp.beta = prior[this.opponentModel.Par[(2*this.evolvingParameterIndex)+2*i]]+dP; // small increase to parameter 1
        var fpdp = inverseSigmoid(this.getAnswerProbability(true,{temp}));
        dfdP = (fpdp-posterior[this.opponentModel.f]) / dP;
        posterior[this.opponentModel.df[i+this.evolvingParameterIndex]]; // store gradient
    }
    return posterior;
}

// Copying the cap from VBA's sigmoid.m
function sigmoid(x) {
    var y = 1 / (1+Math.exp((x*-1)));
    var cap = Math.exp(-4);
    if(y<cap)
        return cap;
    else if(y>(1-cap))
        return 1-cap;
    else
        return y;
}

// invsigmoid.m
function inverseSigmoid(x) {
    var y = Math.log(x/(1-x));
    var cap = 9.2102;
    if(y>cap)
        return cap;
    else if(y<(cap*-1))
        return cap*-1;
    else
        return y;
}

function showDebugInfo(varnames,divID) {
    return;
    var vars = [];
    var out = [];
    if (typeof(varnames) == "undefined" || varnames == "") {
        vars = ["pSELF"];
    } else if (typeof(varnames)=="string") {
        vars = [varnames];
    } else {
        vars = varnames;
    }
    vars.unshift('Trial');
    out[0] = vars;
    for(n=0;n<this.timeSteps.length-1;n++) { // TimeStep
        out[n+1] = [];
        for (i=0;i<vars.length;i++) { // Variable
            if (i==0)
                out[n+1][i] = n;
            else
                out[n+1][i] = this.timeSteps[n][vars[i]];
        }
    }
    drawChart(out,divID);
}
