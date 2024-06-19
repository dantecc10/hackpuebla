<?php
//echo (isset($_POST['name']) && isset($_POST['last_names'])) ? ("Hola " . $_POST['name'] . " " . $_POST['last_names']) : 'No proporcionaste suficientes datos.';

if (
    isset($_POST['auth']) &&
    isset($_POST['id']) &&
    isset($_POST['header']) &&
    isset($_POST['state'])
) {
    include('connection.php');
    $auth = $_POST['auth'];
    $id = $_POST['id'];
    $state = $_POST['state'];
    $header = $_POST['header'];
    if ($auth == '54dsnv8s7hghdf4i7bdgsuASDLHs') {
        $sql = "INSERT INTO `news` VALUES('', $id, '$header', '$state');";
        if ($conn->query($sql) === TRUE) {
            echo ("Success");
            //echo "Noticia creada correctamente";
        } else {
            echo ("Error");
            //echo "Error: " . $sql . "<br>" . $conn->error;
        }
    } else {
        echo ("Fatal");
    }
} else {
    echo ("Fatal");
}
