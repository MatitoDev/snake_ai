module dev.matito.snake_ai {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires net.synedra.validatorfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
    requires eu.hansolo.tilesfx;
    requires com.almasb.fxgl.all;
	requires jdk.httpserver;

	opens dev.matito.snake_ai to javafx.fxml;
    exports dev.matito.snake_ai;
	exports dev.matito.snake_ai.dashboard;
	opens dev.matito.snake_ai.dashboard to javafx.fxml;
}