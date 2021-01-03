var ball = document.getElementById("ball");
var svg_1 = document.getElementById("svg_1");
var svg_2 = document.getElementById("svg_2");
var svg_3 = document.getElementById("svg_3");

window.addEventListener("scroll", bigFun);

function bigFun(){
    rotateBall();
    dragSvg_1();
    //dragSvg_2();
    //dragSvg_3();
}

function rotateBall(){
    var x = window.pageYOffset/400;
    var y = -(window.pageYOffset*1.2);
    ball.style.transform = "translate3d(0px, " + y + "px, 0px) rotateZ(" + x + "rad)";
}

function dragSvg_1(){
    var x = -document.documentElement.scrollTop/50;
    x = x.toFixed(2);
    var fix;
    var re = new RegExp('^-?\\d+(?:\.\\d{0,' + (fix || -1) + '})?');
    fix = x.toString().match(re)[0];
    if(fix < 10){
        fix = fix.toString();
        var y = svg_1.getAttribute('d');
        var repl = (y.substring(0,10) + fix +" " + y.substring(13,));
        svg_1.setAttribute('d', repl);
    }
    else{
        fix = fix.toString();
        var y = svg_1.getAttribute('d');
        var repl = (y.substring(0,10) + fix + y.substring(13,));
        svg_1.setAttribute('d', repl);
    }
}
//"M 0,0 L 0,-15 L 15,0"
function dragSvg_2(){
    var x = -window.pageYOffset/30;
    x.toPrecision(2);
    var y = svg_2.getAttribute('d');
    var repl = (y.substring(0,12) + x.toString() + y.substring(15,));
    svg_2.setAttribute('d', repl);
}
//"M 10,0 L 10,-10 L 25,0"
function dragSvg_3(){
    var x = -window.pageYOffset/30;
    x.toPrecision(2);
    var y = svg_3.getAttribute('d');
    var repl = (y.substring(0,14) + x.toString() + y.substring(18,));
    svg_3.setAttribute('d', repl);
}//"M 100,0 L 100,-15 L 85,0"

