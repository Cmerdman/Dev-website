var ball = document.getElementById("ball")

window.addEventListener("scroll", bigFun);

function bigFun(){
    rotateBall();
    dragSvg();
    //colorChnge();

}

function rotateBall(){
    var x = window.pageYOffset/1400;
    var y = (window.pageYOffset*0.2);
    ball.style.transform = "translate3d(0px, " + y + "px, 0px) rotateZ(" + x + "rad)";
}


function dragSvg(){
    var x = -window.pageYOffset/40;
    x = x.toFixed(2);
    var fix;
    var re = new RegExp('^-?\\d+(?:\.\\d{0,' + (fix || -1) + '})?');
    fix = x.toString().match(re)[0];
    fix= parseInt(fix)
    for(var i = 1; i < 45; i++){
        var svg = document.getElementById("svg_" + i.toString());
        var y = svg.getAttribute('d');
        if(i< 23){
            var output = -Math.pow((i),1.3) + fix;
            var fixed;
            if(output < -99){
                output = -99;
            }
            var rer =new RegExp('^-?\\d+(?:\.\\d{0,' + (fixed || -1) + '})?');
            fixed = output.toString().match(rer)[0];
        }else{
            var output = -Math.pow((45-i),1.3) +fix;
            var fixed;
            if(output < -99){
                output = -99;
            }
            var rer =new RegExp('^-?\\d+(?:\.\\d{0,' + (fixed || -1) + '})?');
            fixed = output.toString().match(rer)[0];
        }
        var repl = (y.substring(0,15) + fixed +"  " + y.substring(18,));
        svg.setAttribute('d', repl);
    }
}


