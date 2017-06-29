/* Script implements variational bayes equations from Devaine et al. 2014
    Code: Matt Jaquiery (2017)

    Usage:
    >// set up the TOM as the hider
    >tom(tomLevel,ROLES.ROLE_HIDER);
    >// Get the players' answers for the round
    >answers[roundNumber++] = {hider: tom.getNextAnswer(), seeker: Math.round(Math.random())];
    >// Format the answers appropriately for ToM
    >tomAnswers = {self: answers[roundNumber-1].hider, opponent: answers[roundNumber-1].seeker};
    >// get TOM to update its beliefs based on the evidence
    >tom.processFeedback(tomAnswers);
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
    this.name = level+"-ToM";
    this.debug = (debug==true);
    // options.inF (prepare_kToM.m)
    this.game = payoffMatrix; // inF.game
    var evolvingParameterMask = [1,0,0]; // inF.dummyPar
    // End of inG properties [missing properties either ignored or calculated on-the-fly]
    this.inputStructure = this.getInputStructure(level,role,evolvingParameterMask);
    // Non-structural properties:
    this.initialPrior = [0,0,0,0,0,1,-1,0,-1,0,0,0]; // HACK options.priors.muX0, but the original is quite hacky, too
    this.theta = -1*Math.log(2); // theta - prior volatility
    this.phi = [-2,0,1]; // temperature, bias, perseveration
    this.observationParams = this.phi;
    this.evolvingParams = [this.theta];
    this.hiddenStates = [this.evolveHiddenStates(this.initialPrior,this.theta,null,this.inputStructure)]; // initalisze x - hidden state information - 1ToM has 12 variables containing info indexed by this.opponentModel.
}

tom.prototype.getInputStructure = function (level, role, evolvingParameters) {
    var out = {};
    // Properties shared by 0Tom and 1Tom:
    out.level = level;
    out.role = (role==ROLES.SEEKER)? ROLES.SEEKER : ROLES.HIDER;
    out.index = { hiddenState: [0,1] };

    if (level==0)  // Finished with shared properties
        return out;

    // Configue hidden states indexing if we're not 0ToM
    // These indices will identify locations in the hiddenStates list holding paritcular parameters
    out.index = {
        hiddenState: [0,1], // Core structure
        f: 2,
        df: [3,4,5],
        params: [6,7,8,9,10,11],
        Es: [6,8,10], // This and below are just shortcuts
        Vs: [7,9,11],
    };
    out.evolvingParameterMask = evolvingParameters;
    return out;
}

tom.prototype.getNextAnswer = function() {
    //return (Math.random()<this.getAnswerProbability(processingOther))? 1 : 0;
    return (this.getAnswerProbability(this.hiddenStates[this.hiddenStates.length-1],this.observationParams,this.inputStructure)>0.5)?1:0; // Copy of demo_recur's rule
}

// MatLab version g_kToM.m and ObsRecGen.m
// inputStructure: level, role
tom.prototype.getAnswerProbability = function(hiddenState, observationParameters, inputStructure) {
    this.d("[ObsRecGen]");
    var pOP; // Opponent's probability of picking 1
    var Vx;
    var a = 0.36; // ObsRecGen.m line 28

    if(inputStructure.level==0) {
        var Ex = hiddenState[0]; // E[log-odds of P(opponentChoice=1)]
        //this.d("Ex [mx] = "+Ex);
        Vx = Math.exp(hiddenState[1]); // V[log-odds of P(opponentChoice=1)]
        //this.d("Vx = "+Vx);
        pOP = this.sigmoid(Ex / Math.sqrt(1+a*Vx)); // P(opponentChoice=1)
        //this.d("pOP [Po] = "+pOP);
    } else {
        var pK = 1; // assume opponent is 0ToM
        var f = hiddenState[inputStructure.index.f];
        Vx = 0;
        for (var i=0;i<inputStructure.index.df.length;i++) {
            var df = hiddenState[inputStructure.index.df[i]];
            var sigma = Math.exp(hiddenState[inputStructure.index.Vs[i]]); // V[theta] QUESTION: should this be .Par[2*i]? ObsRecGen.m line 64
            Vx = Vx+(sigma*Math.pow(df,2)); // V[x(theta)]
        }
        pOP = this.sigmoid(f/Math.sqrt(1+a*Vx)); // P(opponentChoice=1)
    }
    this.d("pOP [Po] = "+pOP);
    // Now we know pOP we can work out our own answer
    var incentive = this.getIncentive(pOP,Math.exp(observationParameters[0]),inputStructure.role,this.game);
    //this.d("incentive = "+incentive);
    var bias = (observationParameters.length>1)? observationParameters[1] : 0;
    this.d("[ObsRecGen output]: "+this.sigmoid(incentive+bias));
    return this.sigmoid(incentive+bias);
}

// MatLab fplayer.m
// Player incentive function
tom.prototype.getIncentive = function(pOP, beta, role, game) {
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
tom.prototype.processFeedback = function(answers) {
    this.d(this.hiddenStates[this.hiddenStates.length-1]);
    this.hiddenStates[this.hiddenStates.length] = this.evolveHiddenStates(this.hiddenStates[this.hiddenStates.length-1],this.evolvingParams,answers,this.inputStructure);
    this.d("New hiddenStates:");
    this.d(this.hiddenStates[this.hiddenStates.length-1]);
}
    // MatLab version f_kToM.m and RecToMfunction.m
    // answers is either -1|0 (special cases for initialisation) or
    // answers = {opponent: 0|1, choice: 0|1}
    // evolvingParams is 'theta'
tom.prototype.evolveHiddenStates = function(hiddenState,evolvingParams,answers,inputStructure) {
    this.d("[RecToMfunction]");

    if(typeof answers!="object") {
        this.d("answers is not valid:");
        this.d(answers)
        return hiddenState; // If we didn't get answers (e.g. no user response) then don't update hidden states
    }
    // level 0 is goverened by MATLAB file evolution0bisND
    // hiddenState should be [x,y] interpreted as E and V of the log-odds of opponent's choice
    if(inputStructure.level == 0) {
        this.d("[evolution0bisND (theta="+evolvingParams[0]+")] input: "+[hiddenState[0],hiddenState[1]]);
        if(answers==null) {
            this.d("No answer supplied");
            return hiddenState; // No update if no answer given
        }
        var E0 = hiddenState[0]; // current E[log-odds]
        //this.d("E0 = "+E0);
        var V0 = Math.exp(hiddenState[1]); // current V[log-odds]
        //this.d("V0 = "+V0);
        var p0 = this.sigmoid(E0); // current estimate of P(opponentChoice=1)
        //this.d("p0 = "+p0);
        var volatility = Math.exp(evolvingParams[0]); // exp(theta)
        //this.d("volatility = "+volatility);
        var V = 1 / ((1/(volatility+V0)) + p0*(1-p0)); // updated V[log-odds]
        var E = E0 + V*(answers.opponent-p0); // updated E[log-odds] via Laplace-Kalman update rule
        this.d("[evolution0bisND] fx="+[this.inverseSigmoid(this.sigmoid(E)),Math.log(V)]);
        return [this.inverseSigmoid(this.sigmoid(E)),Math.log(V)]; // Packaged for numerical reasons
    }

    var prior = hiddenState.slice(0); // copy hiddenState into a new variable for legibility

    // 1ToM case, back to RecToMfunction.m
    // NOTE: we skip P(k') because we assume k'=0
    var posterior = []; // posterior starts blank
    for (var i=0;i<prior.length;i++) {
        posterior[i] = 0;
    }
    // Update beliefs about opponent's parameters E[theta] and V[theta]
    var pK = 1; // Probability opponent level = 0 see note above
    var pOP = this.sigmoid(prior[inputStructure.index.f]); // P(opponentAnswer=1)
    for (var i=0;i<inputStructure.index.df.length;i++) { // Update all the hidden states tracking opponent parameters
        if(answers==null) {
            // No answer supplied, so just copy parameters from previous round
            posterior[inputStructure.index.Vs[i]] = prior[inputStructure.index.Vs[i]];
            posterior[inputStructure.index.Es[i]] = prior[inputStructure.index.Es[i]];
            continue;
        }
        var df = prior[inputStructure.index.df[i]]; // d[x(theta)]/dtheta
        var V0 = Math.exp(prior[inputStructure.index.Vs[i]]); // prior variance
        if(i<evolvingParams.length) {
            V0 = V0 + Math.exp(evolvingParams[0]); // dilute using theta if necessary
        }
        var Vu = 1 / ((1/V0)+pK*pOP*(1-pOP)*Math.pow(df,2)); // posterior variance
        var E0 = prior[inputStructure.index.Es[i]]; // Prior mean
        var Eu = E0 + pK*Vu*(answers.opponent-pOP)*df; // Posterior mean
        // store values with numerical tweaks
        posterior[inputStructure.index.Vs[i]] = Math.log(Vu);
        posterior[inputStructure.index.Es[i]] = this.inverseSigmoid(this.sigmoid(Eu));
    }
    // Technically that's all we need (formally). But to keep par with the MatLab we augment the hidden states with dummy states derived from the sufficient statistics.

    // Set up some handy reference variables to save lots of looping later
    var priorModelHiddenState = [];
    for (var i=0;i<inputStructure.index.hiddenState.length;i++) {
        priorModelHiddenState[i] = prior[inputStructure.index.hiddenState[i]]; // technically overcomplicated, but will save time if this gets extended later
    }
    var evoParams = this.getParams('evo',posterior,inputStructure);
    var obsParams = this.getParams('obs',posterior,inputStructure);

    // Simulate opponent's hidden states
    this.d("Update opponent's hidden states.");

    var temp = {};
    temp.hiddenState = priorModelHiddenState.slice(0);
    temp.evoParams = evoParams.slice(0);
    temp.iS = this.getInputStructure(inputStructure.level-1, (1-inputStructure.role));
    temp.ans = (answers==null)? null : {self: answers.opponent, opponent: answers.self} // reverse answers for opponent
    var opponentHiddenState = this.evolveHiddenStates(temp.hiddenState,temp.evoParams,temp.ans,temp.iS);
    // Copy the returned states into the posterior. Again, overcomplicated but...
    for (var i=0;i<opponentHiddenState.length;i++) {
        posterior[inputStructure.index.hiddenState[i]] = opponentHiddenState[i];
    }
    this.d("Opponent states updated.");

    // Simulate opponent's f
    this.d("Get new f");
    temp.hiddenState = opponentHiddenState; // use UPDATED hidden state to get new f
    temp.obsParams = this.getParams('obs',posterior,inputStructure); // also use updated obsParams
    this.d(temp);
    posterior[inputStructure.index.f] = this.inverseSigmoid(this.getAnswerProbability(temp.hiddenState,temp.obsParams,temp.iS));
    this.d("f_new = "+posterior[inputStructure.index.f]);
    this.d("New f acquired.");

    // Get derivatives df
    this.d("Get dx wrt evolving params");
    var dfdP = [];
    // find derivatives for each evolving parameter (again, only 1 in practice but extensibility...)
    for (var i=0;i<evoParams.length;i++) { // Loop over all evolve params
        this.d("[Parev(q)] = "+evoParams[i]);
        var dP = 1e-4*evoParams[i]; // small parameter increment
        if(Math.abs(dP)<1e-4)
            dP = 1e-4;
        // copy evolving parameters into temp variable
        temp.evoParams = evoParams.slice(0);
        temp.evoParams[i] = evoParams[i]+dP // small increase in parameter 1
        this.d("Par = "+temp.evoParams[i]);
        temp.hiddenState = priorModelHiddenState;
        var Xpdp = this.evolveHiddenStates(temp.hiddenState,temp.evoParams,temp.ans,temp.iS); // See how hidden states evolve
        temp.hiddenState = Xpdp; // Use the new hidden states to find the derivative
        this.d(temp);
        var fpdp = this.inverseSigmoid(this.getAnswerProbability(temp.hiddenState,temp.obsParams,temp.iS)); // get E(x(P+dP))
        var dfdP = (fpdp - posterior[inputStructure.index.f]) / dP; // Gradient w.r.t param 1
        posterior[inputStructure.index.df[i]] = dfdP;
        this.d("Got dx wrt evoParam("+i+"): "+dfdP);
    }
    this.d("Got dx wrt evolving params.");

    // Derive fx wrt observational parameters
    this.d("Get dx wrt observational params");
    this.d(temp);
    dfdP = [];
    temp.hiddenState = opponentHiddenState; // switch back to the opponent hidden states
    for (var i=0;i<obsParams.length;i++) { // the two observational parameters
        temp.obsParams = this.getParams('obs',posterior,inputStructure);
        var param = temp.obsParams[i];
        this.d("Obs Param "+i+"/2 = "+param);
        var dP = 1e-4*param; // small parameter increment
        if(Math.abs(dP)<1e-4)
            dP = 1e-4;
        temp.obsParams[i] = param+dP; // increment the relevant parameter
        var fpdp = this.inverseSigmoid(this.getAnswerProbability(temp.hiddenState,temp.obsParams,temp.iS));
        this.d("fpdp = "+fpdp);
        dfdP = (fpdp-posterior[inputStructure.index.f]) / dP;
        this.d("dfdP = "+dfdP)
        posterior[inputStructure.index.df[i+evoParams.length]] = dfdP; // store gradient, remember df holds delta-F for both evo and obs params
        this.d("Found dx wrt obsParam["+i+"]: "+dfdP);
    }
    this.d("Got dx wrt obs params.");
    return posterior;
}

// Extract parameters of a particular kind (pType) from hiddenState, as indexed by inputStructure
tom.prototype.getParams = function(pType,hiddenState,inputStructure) {
    var evoParams = [];
    var obsParams = []; // if parameters don't evolve they're 'observational'
    for (var i=0;i<inputStructure.evolvingParameterMask.length;i++) {
        if(inputStructure.evolvingParameterMask[i]==1) { // Only deal with parameters which actually evolve
            evoParams[evoParams.length] = hiddenState[inputStructure.index.params[2*i]];
        } else {
            obsParams[obsParams.length] = hiddenState[inputStructure.index.params[2*i]];
        }
    }
    return (pType.substring(0,1)=="o")? obsParams : evoParams;
}

// Copying the cap from VBA's sigmoid.m
tom.prototype.sigmoid = function(x) {
    var y = 1 / (1+Math.exp((x*-1)));
    var cap = 1e-4;
    if(y<cap)
        return cap;
    else if(y>(1-cap))
        return 1-cap;
    else
        return y;
}

// invsigmoid.m
tom.prototype.inverseSigmoid = function(x) {
    var y = Math.log(x/(1-x));
    var cap = Math.log(1e-4);
    if(y<=cap)
        return cap;
    else if(y>=(cap*-1))
        return cap*-1;
    else
        return y;
}

tom.prototype.d = function (x) {
    if(this.debug) console.log(x)
}

tom.prototype.showDebugInfo = function(varnames,divID) {
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
