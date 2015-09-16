package org.gradle;

import java.io.*;

import javax.servlet.http.*;
import javax.servlet.*;

public class PredictServlet extends HttpServlet {
    public void doGet (HttpServletRequest req, HttpServletResponse res) throws ServletException, IOException {
        PrintWriter out = res.getWriter();
        out.print("Hello Gradle");
        out.close();
    }
}
