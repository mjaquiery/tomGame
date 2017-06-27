/* Script implements variational bayes equations from Devaine et al. 2014
    Code: Matt Jaquiery (2017)
*/
var ROLE_NAMES = {0: "HIDER", 1: "SEEKER"};
var ROLES = {ROLE_HIDER: 0, ROLE_SEEKER: 1};
var TOM_TYPES = {NON_TOM: 0, TOM: 1};

function tom(level,debug) {
    // Functions
    this.level = level;
    this.debug = (debug==true);
    this.init = init;
    this.getNextAnswer = getNextAnswer;
    this.giveFeedback = giveFeedback;
    this.showDebugInfo = showDebugInfo;
}

// timeSteps is a time-index listing known values of all variables for a given time point
    /* each timestep is an object with the following parameters:
    {
    aOP - answer given by opponent - set during giveFeedback
    aSELF - answer give by self- set during getNextAnswer
    pOP - probability opponent picks option 1 - set during giveFeedback
    pSELF - probability self picks option 1 - set during pSELF
    mu - opponent's log-odds estimate - set during mu
    sigma - opponent's uncertainty about mu - set during sigma
    volatility - opponent's learning rate volatility - static
    beta - noisiness of opponent's behaviour - static
    }
    */
function init(role) {
    if(this.debug)
        console.log("Initializing ToM")
    // Parameters
    this.tIndex = 0;
    this.timeSteps = [];
    this.timeSteps[this.tIndex] = {
        beta: 0.50,
        pOP: 0.5,
        volatility: 0.05,
        mu: 0.5,
        sigma: 1.0,
    };
    this.role = role;
    this.opponentRole = (role==1)? 0 : 1;
    this.type = TOM_TYPES.TOM;
    this.name = "ToM"+this.level;
}

function getNextAnswer() {
    this.timeSteps[this.tIndex].pSELF = P(this); // Equation 1.i
    this.timeSteps[this.tIndex].aSELF = (Math.random() < this.timeSteps[this.tIndex].pSELF)? 1:0;
    //if (this.role)
    //    this.timeSteps[this.tIndex].aSELF = (this.timeSteps[this.tIndex].aSELF == 1)? 0 : 1;
    return this.timeSteps[this.tIndex].aSELF;
}

function giveFeedback(choice) {
    this.timeSteps[this.tIndex].aOP = choice; // record the choice
    this.timeSteps[this.tIndex].mu = get_mu(this);
    this.tIndex++; // move on to the next time step
    this.timeSteps[this.tIndex] = {
        beta: this.timeSteps[this.tIndex-1].beta,
        volatility: this.timeSteps[this.tIndex-1].volatility
    };
    if (this.level == 1) {
        this.timeSteps[this.tIndex].pOP = sigmoid(v(this)); // Equation 4.i
    } else {
        this.timeSteps[this.tIndex].pOP = get_pOP0(this); // Equation 3.i
    }
    if(this.debug)
        console.log("Choice "+choice+">"+this.name+"("+this.role+") pOP = "+this.timeSteps[this.tIndex].pOP);
}

// Equation 1.i
// beta is the exploration temperature which controls magnitude of behavioural noise (i.e. a sensitivity parameter)
// pOP is the probability opponent chooses option 1
function P(obj) {
    var timeIndex = obj.tIndex;
    var pOP = obj.timeSteps[timeIndex].pOP;
    var beta = obj.timeSteps[timeIndex].beta;
    var v = V(1,obj) - V(0,obj);
    return sigmoid(v/beta);
}

// Equation 1.ii
// i is the provisional choice we are testing
// pOP is the probability opponent chooses option 1
function V(i, obj) {
    var pOP = obj.timeSteps[obj.tIndex].pOP;
    var op1 = pOP * payoffMatrix[i][1][obj.role];
    var op0 = (1-pOP) * payoffMatrix[i][0][obj.role];
    return op1 + op0;
}

// Equation 3.ii
function get_mu(obj,skipAssignVars) {
    var timeIndex = obj.tIndex;
    var last_mu = (timeIndex==0)? obj.timeSteps[timeIndex].mu : obj.timeSteps[timeIndex-1].mu;
    var sigma = get_sigma(obj);
    var aOP = obj.timeSteps[timeIndex].aOP;
    var x = sigmoid(last_mu);
    x = aOP - x;
    x = sigma * x;
    var out = last_mu + x;
    if (!skipAssignVars)
        obj.timeSteps[timeIndex].mu = out;
    return out;
}

// Equation 3.iii
function get_sigma(obj,skipAssignVars) {
    var timeIndex = obj.tIndex;
    var last_sigma = (timeIndex==0)? obj.timeSteps[timeIndex].sigma : obj.timeSteps[timeIndex-1].sigma;
    var volatility = obj.timeSteps[timeIndex].volatility;
    var last_mu = (timeIndex==0)? obj.timeSteps[timeIndex].mu : obj.timeSteps[timeIndex-1].mu;
    var x = 1 / (last_sigma + volatility);
    var y = sigmoid(last_mu);
    y = y * (1-y);
    var out = 1 / (x+y);
    if (!skipAssignVars)
        obj.timeSteps[timeIndex].sigma = out;
    return out;
}

// Equation 4.ii
// pSELF is self's guess at opponent's prediction of self's next move
function v(obj) {
    var timeIndex = obj.tIndex;
    var pSELF_GUESS = get_pSELF_GUESS(obj);
    var beta = obj.timeSteps[timeIndex].beta;
    var deltaU1 = payoffMatrix[1][1][obj.opponentRole] - payoffMatrix[1][0][obj.opponentRole];
    var deltaU0 = payoffMatrix[0][1][obj.opponentRole] - payoffMatrix[0][0][obj.opponentRole];
    var x = pSELF_GUESS * deltaU1;
    var y = (1-pSELF_GUESS) * deltaU0;
    return (x+y)/beta;
}

// Equation 4.iii
function get_pSELF_GUESS(obj,skipAssignVars) {
    var timeIndex = obj.tIndex;
    var last_mu = obj.timeSteps[timeIndex-1].mu;
    var last_sigma = obj.timeSteps[timeIndex-1].sigma;
    var volatility = obj.timeSteps[timeIndex].volatility;
    var x = last_sigma + volatility;
    x = x * 3;
    x = x / Math.pow(Math.PI, 2);
    x = Math.sqrt(1+x);
    x = last_mu / x;
    var out = sigmoid(x);
    if (!skipAssignVars)
        obj.timeSteps[timeIndex].pSELF_GUESS = out;
    return out;
}

// Equation 3.i
function get_pOP0(obj) {
    var timeIndex = obj.tIndex;
    var last_mu = obj.timeSteps[timeIndex-1].mu;
    var last_sigma = obj.timeSteps[timeIndex-1].sigma;
    var volatility = obj.timeSteps[timeIndex].volatility;
    var x = last_sigma+volatility;
    x = x * 3;
    x = x / Math.pow(Math.PI, 2);
    x = Math.sqrt(1+x);
    x = last_mu / x;
    return sigmoid(x);
}

function sigmoid(x) {
    return 1 / (1+Math.exp((x*-1)));
}

var payoffMatrix =  [
                        [[0,1],[1,0]],
                        [[1,0],[0,1]]
                    ]; // h= hider = 0, s=seeker=1. Usage: payoffMatrix[hideChoice][seekChoice].h|s

function showDebugInfo(varnames,divID) {
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
