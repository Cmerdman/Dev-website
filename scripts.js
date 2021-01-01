var ball = document.getElementById("ball");

window.addEventListener("scroll", changeCss);

function rotateBall(){
    ball.style.transform = "rotate("+window.pageYOffset/12 + "deg)";
    ball.style.transform = "tra"
}

function changeCss(){
    var x = window.pageYOffset/400;
    var y = -(window.pageYOffset*1.5);
    ball.style.transform = "translate3d(0px, " + y + "px, 0px) rotateZ(" + x + "rad)";
}


//transform: translate3d(0px, 30.217px, 0px) rotateZ(-0.352601rad);