<?php
echo (isset($_POST['name']) && isset($_POST['last_names'])) ? ("Hola " . $_POST['name'] . " " . $_POST['last_names']) : 'No proporcionaste suficientes datos.';
