function do_ajax()
{
    var req = new XMLHttpRequest();
    var result = document.getElementById('result');
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200)
        {
            result.innerHTML = this.responseText;
        }
        else
        {
            result.innerHTML = "waiting...";
        }
    }
    req.open('POST', '/', true);
    req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
    req.send("name=" + document.getElementById('name').value);
}
function ajax()
{
    var req = new XMLHttpRequest();
    var result = document.getElementById('output');
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200)
        {
            console.log("ajax call success");
            result.innerHTML = 'hello world';
        }
    }
    req.open('GET', '/hello', true);
    req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
    req.send(result.value);
}
function get_translate()
{
    var req = new XMLHttpRequest();
    var result = document.getElementById('translation');
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200)
        {
            console.log("ajax translate call success");
        }
    }
    req.open('GET', '/translate', true);
    req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
    req.send();
}
function get_histogram()
{
    var req = new XMLHttpRequest();
    var result = document.getElementById('histogram');
    req.onreadystatechange = function()
    {
        if(this.readyState == 4 && this.status == 200)
        {
            console.log("ajax histogram call success");
        }
    }
    req.open('GET', '/get_histo', true);
    req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
    req.send();
}
function openCamera()
{
    if (navigator.getUserMedia)
    {
        navigator.getUserMedia(
        {
            video:
            {
            width: 1280,
            height: 700
            },
        audio: false
        },
        function (localMediaStream)
        {
            var video = document.getElementById('camera-stream');
            // url.createObjectURL is deprecated
            // use htmlmediaelement
            video.src = window.URL.createObjectURL(localMediaStream);
        },
        function (err)
        {
            console.log(err.name + ": " + err.message);
        }
        );
    }
    else
    {
        alert("browser does not support getUserMedia");
    }
}
function hello()
{
    alert("hello world");
}